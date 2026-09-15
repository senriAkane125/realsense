"""课题 A：基于 RealSense D455 的 RGB-D 人体状态检测示例（低头检测重写版）。

本文件是 preview_rgbd.py 的副本，只重写了低头检测模块：俯仰角改用 D455
面部点云的法向量计算，不再用 solvePnP 解 6 个近共面关键点的位姿（那种做法
存在镜像解，实测会让俯仰角在 16° 和 100° 之间跳变）。其余功能与原版相同，
方便对照测试。

程序完成的主要功能：
1. 读取 D455 的彩色图像和深度图像，并把两者对齐；
2. 使用 MediaPipe Face Mesh 获取人脸关键点；
3. 使用 EAR（眼睛纵横比）检测眨眼；
4. 用面部点云的法向量估计头部俯仰角（低头为正），判断是否低头；
5. 把面部区域深度拟合成平面，用相对平面的“起伏”区分真人与照片，
   手机/屏幕照片取不到有效深度时同样直接判为照片；
6. 实时显示检测结果，并支持保存实验数据。

这是课程实验原型，不是严格意义上的安全级活体检测系统。实际效果会受到
距离、光照、遮挡、相机角度和深度噪声的影响。
"""

import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Some newer MediaPipe builds keep the legacy Solutions API below
    # mediapipe.python instead of exposing it as mp.solutions.
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import face_mesh as mp_face_mesh


# ---------- 眨眼检测的可调参数 ----------
# 开眼基线取最近这些帧 EAR 的 80% 分位，并限制在下面的范围内。
# 眨眼只是偶发的深谷，不会把 80% 分位拉低，所以基线能稳定反映“睁眼水平”。
BLINK_BASELINE_WINDOW = 150
BLINK_BASELINE_MIN = 0.18
BLINK_BASELINE_MAX = 0.42
# 平均 EAR 低于基线的 75% 视为闭眼；回到基线的 85% 以上视为重新睁开。
BLINK_CLOSE_RATIO = 0.75
BLINK_OPEN_RATIO = 0.85
# 闭眼过程中最低 EAR 需低于基线的 70%，才算“真正闭眼”（过滤轻微抖动）。
BLINK_MIN_DROP_RATIO = 0.70
# 相对最低点回升超过该值，也认为眼睛重新睁开（防止基线估计偏高时漏检）。
BLINK_RECOVER_DELTA = 0.05
# 闭眼持续超过该帧数按“眯眼 / 长时间闭目”处理，不计入眨眼。
BLINK_MAX_CLOSED_FRAMES = 18
# 计数一次眨眼后的冷却帧数，避免同一次眨眼被重复统计。
BLINK_COOLDOWN_FRAMES = 10
# ---------- 低头检测（点云法向量方案）的可调参数 ----------
# 面部区域有效深度占比低于该值时无法估计角度（例如手机屏幕照片）。
HEAD_MIN_VALID_RATIO = 0.40
# 面部点云至少需要这么多点才做平面拟合。
HEAD_MIN_POINTS = 300
# 拟合出的法向量 z 分量绝对值小于该值时，说明脸几乎侧对相机，角度不可靠。
HEAD_MIN_NZ = 0.35
# 角度中值滤波窗口（抑制孤立尖峰）与指数平滑系数（越小越平滑）。
HEAD_MEDIAN_WINDOW = 7
HEAD_SMOOTH_ALPHA = 0.15
# 相对校准姿态超过该角度判为低头。
HEAD_DOWN_ENTER_DEG = 12.0
# 回落到该角度以内才恢复为正常，与上一个值构成迟滞区间，避免临界抖动。
HEAD_DOWN_EXIT_DEG = 6.0
# 状态切换需要连续确认的帧数，抑制单帧误判。
HEAD_CONFIRM_FRAMES = 5
# ---------- 真人/照片判定的可调参数 ----------
# 面部区域有效深度占比低于该值 → 照片。手机/显示器屏幕几乎不反射红外光，
# 真实人脸皮肤则能返回 90% 以上的有效深度，这是最可靠的区分依据。
LIVENESS_MIN_VALID_RATIO = 0.40
# 剔除乱深度时，残差超过“MAD 的该倍数”的点视为噪声（屏幕镜面反射等）。
LIVENESS_MAD_SCALE = 6.0
# 上面的阈值下限：真实人脸本身起伏就有几厘米，不能把鼻尖等正常结构剔除掉。
LIVENESS_MIN_KEEP_BAND_M = 0.03
# 面部区域至少要有这么多有效深度点才做判断。
DEPTH_MIN_SAMPLES = 150
# 点云到拟合平面的起伏（95% 分位 - 5% 分位）超过该值判为真人。
LIVENESS_RELIEF_M = 0.02
# 起伏超过该值说明深度是屏幕镜面反射产生的乱值，不可能是真实人脸。
LIVENESS_MAX_PLAUSIBLE_RELIEF_M = 0.12
# D455 最近稳定工作距离约 0.45 米，更近时深度不可靠，不依据起伏判定。
LIVENESS_MIN_RELIABLE_DEPTH_M = 0.45
# 连续多帧得到相同判断后才切换真人/照片标签，减少单帧噪声造成的跳变。
LIVENESS_STABLE_FRAMES = 5


def distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个二维点之间的欧氏距离。"""
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks: list[np.ndarray], indices: tuple[int, ...]) -> float:
    """计算一只眼睛的 EAR（Eye Aspect Ratio）。

    indices 对应眼睛轮廓上的 6 个 MediaPipe 关键点：
    - p1、p4：眼睛左右两端，用于计算眼睛宽度；
    - p2、p3、p5、p6：上下边缘，用于计算眼睛高度。

    眼睛睁开时上下距离较大，EAR 较大；闭眼时上下距离变小，EAR 较小。
    """
    p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
    horizontal = distance_2d(p1, p4)
    if horizontal < 1e-6:
        return 1.0
    return (distance_2d(p2, p6) + distance_2d(p3, p5)) / (2.0 * horizontal)


class BlinkDetector:
    """基于 EAR 的眨眼检测器（自适应开眼基线）。

    相比固定阈值，这里先估计“开眼基线”：取最近 BLINK_BASELINE_WINDOW 帧
    里 EAR 的 80% 分位。眨眼只是偶发的深谷，不会把 80% 分位拉低，因此基线
    能稳定反映当前用户的睁眼水平，自动适应眼睛大小、戴眼镜、离相机远近等
    差异，避免固定阈值偏高（漏检眨眼）或偏低（误计数）的问题。

    一次眨眼需要同时满足：
    1. 平均 EAR 跌破基线的 BLINK_CLOSE_RATIO（进入闭眼，允许单帧快眨眼）；
    2. 闭眼期间最低 EAR 低于基线的 BLINK_MIN_DROP_RATIO（确保是真闭眼）；
    3. 随后 EAR 回到开眼基线以上，或相对最低点明显回升；
    4. 闭眼总时长不超过 BLINK_MAX_CLOSED_FRAMES 帧（眯眼、闭目不计数）；
    5. 距上次计数已超过冷却帧数（同一次眨眼不会被重复统计）。
    """

    def __init__(self) -> None:
        self.count = 0
        self.ear_samples = deque(maxlen=BLINK_BASELINE_WINDOW)
        self.cooldown = 0
        self.lowest_ear = 1.0
        self.candidate_frames = 0
        self.session_frames = 0

    @property
    def baseline(self) -> float:
        """当前的开眼基线（EAR）：最近样本的 80% 分位并限制在合理范围。"""
        if len(self.ear_samples) < 15:
            return 0.30  # 样本还不够时的经验值
        value = float(np.percentile(self.ear_samples, 80))
        return float(np.clip(value, BLINK_BASELINE_MIN, BLINK_BASELINE_MAX))

    def update(self, left_ear: float, right_ear: float) -> None:
        """输入一帧的左右眼 EAR，更新眨眼计数。"""
        ear = (left_ear + right_ear) / 2.0
        self.ear_samples.append(ear)
        if self.cooldown > 0:
            self.cooldown -= 1

        base = self.baseline
        open_level = base * BLINK_OPEN_RATIO
        close_level = base * BLINK_CLOSE_RATIO

        if ear <= open_level:
            # 只要还没有明确回到开眼水平，就持续累计本次“闭眼会话”。
            self.session_frames += 1

        if ear < close_level:
            # 闭眼（或继续停留在闭眼状态）。
            if self.candidate_frames == 0:
                self.lowest_ear = ear
                self.candidate_frames = 1
            else:
                self.candidate_frames += 1
                self.lowest_ear = min(self.lowest_ear, ear)
        elif self.candidate_frames > 0:
            # 已从最深处回升：判断本次是不是一次合格的眨眼。
            recovered = (
                ear > open_level or ear > self.lowest_ear + BLINK_RECOVER_DELTA
            )
            if recovered:
                if (
                    self.cooldown == 0
                    and self.candidate_frames <= BLINK_MAX_CLOSED_FRAMES
                    and self.session_frames <= BLINK_MAX_CLOSED_FRAMES
                    and self.lowest_ear < base * BLINK_MIN_DROP_RATIO
                ):
                    self.count += 1
                    self.cooldown = BLINK_COOLDOWN_FRAMES
                self.candidate_frames = 0

        if self.candidate_frames > BLINK_MAX_CLOSED_FRAMES:
            # 一直闭着没有回升（眯眼、闭目、照片）：放弃本次候选。
            self.candidate_frames = 0

        if ear > open_level:
            # 眼睛明确睁开，结束本次闭眼会话。
            self.session_frames = 0

    def reset(self) -> None:
        """人脸丢失时重置状态；开眼基线保留，继续适应当前用户。"""
        self.candidate_frames = 0
        self.session_frames = 0


def landmark_pixels(face_landmarks, pwidth: int, height: int) -> list[np.ndarray]:
    """把 MediaPipe 的归一化坐标转换为图像像素坐标。

    MediaPipe 返回的 point.x、point.y 范围通常是 0 到 1，
    需要分别乘以图像宽度和高度，才能在 OpenCV 图像上绘制或读取深度。
    np.clip 用于防止边界点转换后超出图像范围。
    """
    return [
        np.array(
            [
                np.clip(point.x * pwidth, 0, pwidth - 1),
                np.clip(point.y * height, 0, height - 1),
            ],
            dtype=np.float32,
        )
        for point in face_landmarks.landmark
    ]


def _plane_relief(points: np.ndarray) -> Optional[float]:
    """拟合平面，返回点云到平面的垂直距离起伏（95% 分位 - 5% 分位），单位米。

    先用 MAD（中位数绝对偏差）剔除少量乱深度，再重新拟合一次。这样屏幕
    镜面反射产生的个别错误深度不会把起伏值抬高，同时保留鼻尖、下颌这些
    真实面部结构。
    """
    keep = None
    residual = None
    for _ in range(2):
        centered = points - points.mean(axis=0)
        # SVD 的最右奇异向量就是点云的法向量，残差即各点到平面的垂直距离。
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        residual = centered @ vh[2]
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        keep = np.abs(residual - center) <= max(LIVENESS_MAD_SCALE * scale,
                                                LIVENESS_MIN_KEEP_BAND_M)
        if keep.all() or int(keep.sum()) < DEPTH_MIN_SAMPLES:
            break
        points = points[keep]
    if residual is None or keep is None or int(keep.sum()) < DEPTH_MIN_SAMPLES:
        return None
    kept = residual[keep]
    return float(np.percentile(kept, 95) - np.percentile(kept, 5))


def face_depth_relief(
    depth_image: np.ndarray,
    pixels: list[np.ndarray],
    depth_scale: float,
    camera_matrix: np.ndarray,
) -> tuple[Optional[float], Optional[float]]:
    """把面部区域深度反投影成三维点云，返回（点云起伏值, 面部平均距离），单位米。

    步骤：
    1. 用全部人脸关键点生成面部区域掩膜，并收缩边缘，避免混入头发和背景；
    2. 统计有效深度占比：手机/显示器屏幕几乎不反射红外光，该占比会很低；
    3. 用相机内参把像素反投影成三维点：X=(u-cx)*z/fx，Y=(v-cy)*z/fy，Z=z；
    4. 拟合最优平面并计算各点到平面的垂直距离，剔除乱深度后取
       “95% 分位 - 5% 分位”作为起伏值。

    真人脸是三维曲面（鼻尖凸出、下颌后收），起伏通常有 2~5 厘米；
    平面照片上的点近似共面，起伏只剩毫米级传感器噪声。
    先拟合平面可以抵消低头、抬头、照片倾斜造成的整体倾斜。

    无法判断时返回 (None, None)，例如面部区域几乎没有有效深度。
    """
    face_points = np.array(pixels, dtype=np.int32)
    mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(face_points), 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None
    raw = depth_image[ys, xs].astype(np.float64) * depth_scale
    valid = raw > 0
    # 有效深度占比过低 → 该区域不反射红外光（手机/显示器屏幕照片）。
    if valid.mean() < LIVENESS_MIN_VALID_RATIO:
        return None, None
    ys, xs, depths = (
        ys[valid].astype(np.float64),
        xs[valid].astype(np.float64),
        raw[valid],
    )

    median_depth = float(np.median(depths))
    if depths.size < DEPTH_MIN_SAMPLES:
        return None, None

    # 反投影成三维点云（课题要求的“点云”信息在这里用到）。
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(
        [(xs - cx) * depths / fx, (ys - cy) * depths / fy, depths]
    )

    # 反投影得到的点云拟合平面，起伏值即“脸部曲面偏离平面的程度”。
    relief = _plane_relief(points)
    return relief, median_depth


def judge_liveness(relief: Optional[float], face_distance: Optional[float]) -> str:
    """给出课题要求的二分类结果："真人" 或 "照片"。"""
    if relief is None or face_distance is None:
        return "照片"  # 面部取不到有效深度：手机/屏幕照片
    if relief > LIVENESS_MAX_PLAUSIBLE_RELIEF_M:
        return "照片"  # 起伏离谱，是屏幕镜面反射产生的乱深度
    if face_distance < LIVENESS_MIN_RELIABLE_DEPTH_M:
        # 距离小于 D455 最近工作距离时深度不可靠，此时能返回完整深度
        # 说明表面是漫反射的皮肤，而不是会把红外光反射走的屏幕。
        return "真人"
    return "真人" if relief >= LIVENESS_RELIEF_M else "照片"


def face_normal_from_depth(
    depth_image: np.ndarray,
    pixels: list[np.ndarray],
    depth_scale: float,
    camera_matrix: np.ndarray,
) -> Optional[np.ndarray]:
    """用面部点云拟合平面，返回指向相机的单位法向量。

    这是重写后的俯仰角来源，思路和“真人/照片”判定一致，都是把面部深度
    反投影成三维点云后再做平面拟合：

    1. 用人脸关键点生成面部掩膜，取掩膜内的有效深度像素；
    2. 用相机内参反投影成三维点 X=(u-cx)*z/fx，Y=(v-cy)*z/fy，Z=z；
    3. SVD 拟合最优平面，最小奇异值对应的方向即法向量；
    4. 用 MAD 剔除乱深度后重新拟合一次；
    5. 统一方向：让法向量指向相机（z 分量为负）。

    为什么不用 solvePnP：那 6 个关键点近似共面，位姿求解存在镜像解，
    实测中俯仰角会在 16° 和 100° 之间突然翻转。点云法向量是对整个面部
    区域平均的结果，数值连续、没有分支跳变，而且不依赖人脸三维模型。

    返回 None 表示该帧无法估计角度（深度不足、遮挡、或脸几乎侧对相机）。
    """
    face_points = np.array(pixels, dtype=np.int32)
    mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(face_points), 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    depths = depth_image[ys, xs].astype(np.float64) * depth_scale
    valid = depths > 0
    if float(valid.mean()) < HEAD_MIN_VALID_RATIO:
        return None
    ys, xs, depths = (
        ys[valid].astype(np.float64),
        xs[valid].astype(np.float64),
        depths[valid],
    )
    if depths.size < HEAD_MIN_POINTS:
        return None

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(
        [(xs - cx) * depths / fx, (ys - cy) * depths / fy, depths]
    )

    normal = None
    for _ in range(2):
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2]
        residual = centered @ normal
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        keep = np.abs(residual - center) <= max(6.0 * scale, 0.03)
        if keep.all() or int(keep.sum()) < HEAD_MIN_POINTS:
            break
        points = points[keep]

    if normal is None:
        return None
    if normal[2] > 0:  # 相机 z 轴指向正前方，人脸法向量应指向相机（z 为负）
        normal = -normal
    if abs(float(normal[2])) < HEAD_MIN_NZ:
        return None  # 脸几乎侧对相机，法向量不可靠
    return normal


def face_pitch_from_depth(
    depth_image: np.ndarray,
    pixels: list[np.ndarray],
    depth_scale: float,
    camera_matrix: np.ndarray,
) -> Optional[float]:
    """返回头部俯仰角（度）：正视约 0°，低头为正，抬头为负。

    相机坐标系的 y 轴朝下，所以法向量的 y 分量越大表示脸越朝下。
    """
    normal = face_normal_from_depth(depth_image, pixels, depth_scale, camera_matrix)
    if normal is None:
        return None
    return float(np.degrees(np.arctan2(normal[1], -normal[2])))


class HeadPoseDetector:
    """低头检测：先平滑俯仰角，再用迟滞区间 + 连续帧确认输出状态。

    俯仰角由面部点云法向量给出（数值连续、无镜像解跳变），这一层负责把它
    变成稳定的“正常/低头”状态：
    1. 中值滤波：吃掉单帧或两帧的异常角度；
    2. 指数平滑：抑制抖动，同时保留持续几十帧的真实低头动作；
    3. 迟滞区间：超过 HEAD_DOWN_ENTER_DEG 才算低头，回落到
       HEAD_DOWN_EXIT_DEG 以内才算恢复，避免角度停在阈值附近来回跳；
    4. 连续帧确认：目标状态连续 HEAD_CONFIRM_FRAMES 帧成立才真正切换。
    """

    def __init__(self) -> None:
        self.raw: Optional[float] = None
        self.smoothed: Optional[float] = None
        self.neutral: Optional[float] = None
        self.state = "正常"
        self.pending: Optional[str] = None
        self.pending_frames = 0
        self.raw_window = deque(maxlen=HEAD_MEDIAN_WINDOW)

    def update(self, pitch: Optional[float]) -> str:
        """输入本帧俯仰角，返回当前头部状态文字。"""
        self.raw = pitch
        if pitch is None:
            self.smoothed = None
            self.raw_window.clear()
            return "--"
        # 1) 中值滤波：抑制单帧或两帧的异常角度尖峰。
        self.raw_window.append(pitch)
        filtered = float(np.median(self.raw_window))
        # 2) 指数平滑：让曲线变缓，但保留低头这种持续几十帧的姿态变化。
        if self.smoothed is None:
            self.smoothed = filtered
        else:
            self.smoothed += HEAD_SMOOTH_ALPHA * (filtered - self.smoothed)
        if self.neutral is None:
            return "待校准（按 N 键）"

        delta = self.smoothed - self.neutral
        if delta > HEAD_DOWN_ENTER_DEG:
            target = "低头"
        elif delta < HEAD_DOWN_EXIT_DEG:
            target = "正常"
        else:
            target = self.state  # 迟滞区间内保持原状态
        self.state = self._confirm(target)
        return self.state

    def _confirm(self, target: str) -> str:
        """目标状态连续确认足够帧后才真正切换。"""
        if target == self.state:
            self.pending = None
            self.pending_frames = 0
            return self.state
        if target == self.pending:
            self.pending_frames += 1
        else:
            self.pending = target
            self.pending_frames = 1
        if self.pending_frames >= HEAD_CONFIRM_FRAMES:
            self.state = target
            self.pending = None
            self.pending_frames = 0
        return self.state

    def calibrate(self) -> Optional[float]:
        """把当前平滑后的俯仰角记为“正常姿态”。"""
        if self.smoothed is not None:
            self.neutral = self.smoothed
        return self.neutral

    def reset(self) -> None:
        """人脸丢失时清空平滑值，但保留校准结果。"""
        self.raw = None
        self.smoothed = None
        self.raw_window.clear()
        self.pending = None
        self.pending_frames = 0


# ---------- 中文文字绘制 ----------
# OpenCV 自带字体不支持中文，这里用 Pillow + 系统中文字体渲染，
# 渲染结果按文字内容缓存，避免每帧重复生成。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",  # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
)
_FONT = None
_LABEL_CACHE: dict = {}


def _label_font():
    """加载第一个可用中文字体，只加载一次。"""
    global _FONT
    if _FONT is None:
        for path in _FONT_CANDIDATES:
            if Path(path).exists():
                _FONT = ImageFont.truetype(path, 22)
                break
    return _FONT


def put_label_cn(image: np.ndarray, text: str, y: int, color=(0, 255, 0)) -> None:
    """在图像 (15, y) 位置绘制一行带黑色描边的中文文字（color 为 BGR）。"""
    font = _label_font()
    if font is None:
        return
    key = (text, color)
    rgba = _LABEL_CACHE.get(key)
    if rgba is None:
        left, top, right, bottom = font.getbbox(text)
        canvas = Image.new(
            "RGBA", (right - left + 16, bottom - top + 16), (0, 0, 0, 0)
        )
        drawer = ImageDraw.Draw(canvas)
        drawer.text(
            (8 - left, 8 - top), text, font=font,
            fill=(color[2], color[1], color[0], 255),
            stroke_width=2, stroke_fill=(0, 0, 0, 255),
        )
        rgba = np.array(canvas)
        if len(_LABEL_CACHE) > 400:
            _LABEL_CACHE.clear()
        _LABEL_CACHE[key] = rgba
    height, width = rgba.shape[:2]
    if y + height > image.shape[0] or 15 + width > image.shape[1]:
        return
    roi = image[y:y + height, 15:15 + width]
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (
        roi * (1.0 - alpha) + rgba[:, :, :3].astype(np.float32) * alpha
    ).astype(np.uint8)


def main() -> None:
    """启动相机、执行逐帧检测，并显示/保存实验结果。"""
    # pipeline 负责从 RealSense 设备持续获取帧，config 用于配置数据流。
    pipeline = rs.pipeline()
    config = rs.config()

    # 彩色流使用 BGR 格式，便于直接交给 OpenCV；深度流使用 z16 原始深度格式。
    # 640x480、30 FPS 足以完成课程演示，同时计算量相对较小。
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        profile = pipeline.start(config)
    except RuntimeError as error:
        print("无法启动 RealSense 相机。")
        print("请确认 D455 插在 USB 3 接口上，并先关闭 RealSense Viewer。")
        print(f"详细错误：{error}")
        sys.exit(1)

    # 深度相机和彩色相机的光心不同，原始两张图的像素并不一一对应。
    # 对齐到 color 后，彩色图上的人脸关键点可以直接索引 depth_image。
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 从设备读取彩色相机内参：用于面部点云反投影和目标姿态/活体判定。
    color_intrinsics = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    )
    camera_matrix = np.array(
        [
            [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
            [0.0, color_intrinsics.fy, color_intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64,
    )

    # 保存目录使用相对路径，运行程序后会出现在当前工作目录下的 captures 文件夹。
    output_dir = Path("captures")

    # 以下变量是跨帧状态：眨眼计数、正常头部姿态、活体判断稳定计数等。
    previous_timestamp_ms = None
    blink = BlinkDetector()
    head = HeadPoseDetector()
    # 活体结果只取“真人/照片”两种；未检测到人脸时为 None（界面显示 --）。
    liveness_label: Optional[str] = None
    real_counter = 0
    photo_counter = 0

    print(f"已启动 D455，深度比例：{depth_scale:.6f} 米/单位")
    print("快捷键：Q/Esc 退出 | S 保存截图 | N 校准正常头部姿态")

    try:
        # FaceMesh 对每一帧图像输出人脸关键点。
        # refine_landmarks=True 会提供更细致的眼睛和嘴部关键点。
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                # 等待一组新的彩色帧和深度帧。
                frames = pipeline.wait_for_frames()

                # 先做 RGB-D 对齐，再分别取出彩色帧和深度帧。
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # RealSense 帧对象转换为 NumPy 数组后，才能使用 OpenCV/NumPy 处理。
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # 深度图是单通道距离数据，转换成伪彩色图只是为了方便人眼观察，
                # 算法实际使用的仍然是上面的原始 depth_image。
                depth_color = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                # 使用相邻帧的时间戳估算当前 FPS。
                timestamp_ms = color_frame.get_timestamp()
                fps = 0.0
                if previous_timestamp_ms is not None and timestamp_ms > previous_timestamp_ms:
                    fps = 1000.0 / (timestamp_ms - previous_timestamp_ms)
                previous_timestamp_ms = timestamp_ms

                # MediaPipe 使用 RGB，而 OpenCV 默认是 BGR，所以这里先转换颜色通道。
                result = face_mesh.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                face_status = "未检测"
                head_status = "--"
                ear_value = 0.0
                pitch_value = None
                depth_relief: Optional[float] = None

                if result.multi_face_landmarks:
                    face_landmarks = result.multi_face_landmarks[0]

                    # 当前只处理画面中置信度最高/返回的第一张人脸。
                    pixels = landmark_pixels(
                        face_landmarks, color_image.shape[1], color_image.shape[0]
                    )
                    # 绘制人脸网格，便于观察关键点检测是否稳定。
                    mp_drawing.draw_landmarks(
                        color_image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 200, 255), thickness=1, circle_radius=1
                        ),
                    )
                    # 用所有关键点的最小/最大坐标生成一个简单人脸外接框。
                    x_values = [point[0] for point in pixels]
                    y_values = [point[1] for point in pixels]
                    cv2.rectangle(
                        color_image,
                        (int(min(x_values)), int(min(y_values))),
                        (int(max(x_values)), int(max(y_values))),
                        (255, 180, 0), 2,
                    )

                    # 左右眼分别计算 EAR；显示用平均值，判定时要求双眼同时闭合。
                    left_ear = eye_aspect_ratio(pixels, (33, 160, 158, 133, 153, 144))
                    right_ear = eye_aspect_ratio(pixels, (263, 387, 385, 362, 380, 373))
                    ear_value = (left_ear + right_ear) / 2.0
                    blink.update(left_ear, right_ear)

                    # 用面部点云法向量估计俯仰角，再交给 HeadPoseDetector 判定。
                    pitch_value = face_pitch_from_depth(
                        depth_image, pixels, depth_scale, camera_matrix
                    )
                    if pitch_value is not None:
                        head_status = head.update(pitch_value)

                    # 真人脸是三维曲面，平面照片是平的，屏幕照片取不到有效深度。
                    depth_relief, face_distance = face_depth_relief(
                        depth_image, pixels, depth_scale, camera_matrix
                    )
                    verdict = judge_liveness(depth_relief, face_distance)
                    if liveness_label is None:
                        # 第一次判断直接采用，界面不出现中间状态。
                        liveness_label = verdict
                        real_counter = 0
                        photo_counter = 0
                    elif verdict == liveness_label:
                        real_counter = 0
                        photo_counter = 0
                    elif verdict == "真人":
                        # 连续多帧确认为真人后才切换，抵消单帧噪声。
                        real_counter += 1
                        photo_counter = 0
                        if real_counter >= LIVENESS_STABLE_FRAMES:
                            liveness_label = "真人"
                            real_counter = 0
                    else:
                        photo_counter += 1
                        real_counter = 0
                        if photo_counter >= LIVENESS_STABLE_FRAMES:
                            liveness_label = "照片"
                            photo_counter = 0
                    face_status = "已检测"
                else:
                    blink.reset()
                    head.reset()
                    real_counter = 0
                    photo_counter = 0
                    liveness_label = None

                # 读取画面中心点的距离，用于展示深度流是否正常工作。
                center_x = depth_frame.get_width() // 2
                center_y = depth_frame.get_height() // 2
                center_depth_m = depth_frame.get_distance(center_x, center_y)
                cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
                shown_pitch = head.smoothed
                if shown_pitch is None:
                    pitch_text = "--"
                elif head.raw is None:
                    pitch_text = f"{shown_pitch:.1f} 度"
                else:
                    pitch_text = f"{shown_pitch:.1f} 度（原始 {head.raw:.1f}）"
                relief_text = "--" if depth_relief is None else f"{depth_relief:.3f} 米"
                liveness_text = liveness_label if liveness_label is not None else "--"
                put_label_cn(color_image, f"人脸：{face_status}", 12)
                put_label_cn(color_image, f"活体：{liveness_text}", 44)
                put_label_cn(
                    color_image,
                    f"眨眼次数：{blink.count}  EAR：{ear_value:.2f}（基线 {blink.baseline:.2f}）",
                    76,
                )
                put_label_cn(color_image, f"头部：{head_status}", 108)
                put_label_cn(color_image, f"俯仰角：{pitch_text}   面部起伏：{relief_text}", 140)
                put_label_cn(color_image, f"中心深度：{center_depth_m:.2f} 米   帧率：{fps:.1f} FPS", 172)
                # 左侧显示带检测结果的彩色图，右侧显示深度伪彩图。
                display = np.hstack((color_image, depth_color))
                # 窗口标题用 ASCII：OpenCV 的 HighGUI 标题栏显示中文会乱码。
                cv2.imshow("Project A - D455 RGB-D", display)

                # waitKey 同时负责刷新窗口和读取键盘输入。
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    # 用户保持正常正视姿态按 N，把平滑后的当前角度作为基准。
                    neutral = head.calibrate()
                    if neutral is not None:
                        print(f"已记录正常头部俯仰角：{neutral:.2f} 度")
                elif key == ord("s"):
                    # 保存原始彩色图、原始深度图和便于答辩展示的拼接预览图。
                    output_dir.mkdir(exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(output_dir / f"{stamp}_color.png"), color_image)
                    cv2.imwrite(str(output_dir / f"{stamp}_depth.png"), depth_image)
                    cv2.imwrite(str(output_dir / f"{stamp}_preview.png"), display)
                    print(f"已保存 RGB-D 数据：{stamp}")
                elif key in (ord("q"), 27):
                    # Q 或 Esc 退出主循环。
                    break
    finally:
        # 无论正常退出还是运行中发生异常，都要释放相机和 OpenCV 窗口。
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
