"""课题 A：基于 RealSense D455 的 RGB-D 人体状态检测示例（头部姿态重写版）。

本文件是 preview_rgbd.py 的副本，头部姿态检测改用 D455 面部点云的法向量
计算俯仰角，不再用 solvePnP 解 6 个近共面关键点的位姿（那种做法存在镜像
解，实测会让俯仰角在 16° 和 100° 之间跳变）。

程序完成的主要功能：
1. 读取 D455 的彩色图像和深度图像，并把两者对齐；
2. 使用 MediaPipe Face Mesh 获取人脸关键点；
3. 使用 EAR（眼睛纵横比）检测眨眼；
4. 用面部点云的法向量估计头部俯仰角（低头为正），判断正常/低头/抬头；
5. 把面部区域深度拟合成平面，用相对平面的“起伏”区分真人与照片，
   手机/屏幕照片取不到有效深度时同样直接判为照片；
6. 实时显示检测结果、低头报警、RGB-D 截图和结果视频；
7. 保存事件日志和会话统计，方便写实验报告。

这是课程实验原型，不是严格意义上的安全级活体检测系统。实际效果会受到
距离、光照、遮挡、相机角度和深度噪声的影响。
"""

import csv
import sys
import time
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
# 平均 EAR 低于基线的 80% 视为闭眼；回到基线的 85% 以上视为重新睁开。
# 10~15 FPS 下快眨眼常只采到一帧浅闭合，比值放宽到 0.80 可减少漏检。
BLINK_CLOSE_RATIO = 0.80
BLINK_OPEN_RATIO = 0.85
# 闭眼过程中最低 EAR 需低于基线的 78%，才算“真正闭眼”（过滤轻微抖动）。
BLINK_MIN_DROP_RATIO = 0.78
# 相对最低点回升超过该值，也认为眼睛重新睁开（防止基线估计偏高时漏检）。
BLINK_RECOVER_DELTA = 0.04
# 闭眼持续超过该秒数按“眯眼 / 长时间闭目”处理，不计入眨眼。
BLINK_MAX_CLOSED_SECONDS = 1.0
# 计数一次眨眼后的冷却时间（秒），避免低帧率下把连续两次真眨眼漏掉。
BLINK_COOLDOWN_SECONDS = 0.25
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
# 相对校准姿态低于负的该角度判为抬头。
HEAD_UP_ENTER_DEG = 12.0
# 回到负的该角度以内才恢复为正常，与抬头进入阈值构成迟滞区间。
HEAD_UP_EXIT_DEG = 6.0
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
# ---------- 演示增强参数 ----------
# 同时检测两张脸，但动作识别仍只围绕画面中面积最大的主目标进行。
MAX_FACES = 2
# 低头状态连续保持该秒数后触发报警。阈值比状态确认更严格，避免一闪而过。
ALARM_HOLD_SECONDS = 1.5
# 会话统计中保留最近这些 FPS 样本，用于输出中位数和 95 分位帧率。
STATS_WINDOW = 120
# 双眼区域连续多帧满足“明显变暗且纹理很少”时，提示可能存在墨镜等遮挡。
OCCLUSION_HOLD_FRAMES = 4
OCCLUSION_DARK_MEAN = 65.0
OCCLUSION_DARK_RATIO = 0.60
OCCLUSION_MAX_STD = 25.0


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
    """基于 EAR 的眨眼检测器（自适应开眼基线 + 按时间判定）。

    相比固定阈值，这里先估计“开眼基线”：取最近 BLINK_BASELINE_WINDOW 帧
    里 EAR 的 80% 分位。眨眼只是偶发的深谷，不会把 80% 分位拉低，因此基线
    能稳定反映当前用户的睁眼水平，自动适应眼睛大小、戴眼镜、离相机远近等
    差异，避免固定阈值偏高（漏检眨眼）或偏低（误计数）的问题。

    一次眨眼需要同时满足：
    1. 平均 EAR 跌破基线的 BLINK_CLOSE_RATIO（进入闭眼，允许单帧快眨眼）；
    2. 闭眼期间最低 EAR 低于基线的 BLINK_MIN_DROP_RATIO（确保是真闭眼）；
    3. 随后 EAR 回到开眼基线以上，或相对最低点明显回升；
    4. 闭眼总时长不超过 BLINK_MAX_CLOSED_SECONDS 秒（眯眼、闭目不计数）；
    5. 距上次计数已超过 BLINK_COOLDOWN_SECONDS 秒（按真实时间冷却，
       低帧率下不会把冷却期拉长到一秒以上）。
    """

    def __init__(self) -> None:
        self.count = 0
        self.ear_samples = deque(maxlen=BLINK_BASELINE_WINDOW)
        self.cooldown_until = 0.0
        self.lowest_ear = 1.0
        self.candidate_started_at: Optional[float] = None

    @property
    def baseline(self) -> float:
        """当前的开眼基线（EAR）：最近样本的 80% 分位并限制在合理范围。"""
        if len(self.ear_samples) < 15:
            return 0.30  # 样本还不够时的经验值
        value = float(np.percentile(self.ear_samples, 80))
        return float(np.clip(value, BLINK_BASELINE_MIN, BLINK_BASELINE_MAX))

    def update(self, left_ear: float, right_ear: float, now_s: float) -> None:
        """输入一帧的左右眼 EAR 和单调时间戳（秒），更新眨眼计数。"""
        ear = (left_ear + right_ear) / 2.0
        self.ear_samples.append(ear)

        base = self.baseline
        open_level = base * BLINK_OPEN_RATIO
        close_level = base * BLINK_CLOSE_RATIO

        if ear < close_level:
            # 进入或持续闭眼：记录本次闭眼的最低点，暂不计数。
            if self.candidate_started_at is None:
                self.candidate_started_at = now_s
                self.lowest_ear = ear
            else:
                self.lowest_ear = min(self.lowest_ear, ear)
            return

        if self.candidate_started_at is None:
            return

        closed_seconds = now_s - self.candidate_started_at
        # 回到开眼水平，或从最低点明显回升，都算重新睁开。
        recovered = ear > open_level or ear > self.lowest_ear + BLINK_RECOVER_DELTA
        if recovered:
            if (
                now_s >= self.cooldown_until
                and closed_seconds <= BLINK_MAX_CLOSED_SECONDS
                and self.lowest_ear < base * BLINK_MIN_DROP_RATIO
            ):
                self.count += 1
                self.cooldown_until = now_s + BLINK_COOLDOWN_SECONDS
            self.candidate_started_at = None
            self.lowest_ear = 1.0
        elif closed_seconds > BLINK_MAX_CLOSED_SECONDS:
            # 眼睛没有明显睁开且闭得太久：按眯眼/闭目放弃本次候选。
            self.candidate_started_at = None
            self.lowest_ear = 1.0

    def reset(self) -> None:
        """人脸丢失或眼部遮挡时重置状态；开眼基线保留，继续适应当前用户。"""
        self.candidate_started_at = None
        self.lowest_ear = 1.0


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
) -> tuple[Optional[float], Optional[float], float]:
    """把面部区域深度反投影成三维点云，返回（点云起伏值, 面部距离, 有效深度占比）。

    步骤：
    1. 用全部人脸关键点生成面部区域掩膜，并收缩边缘，避免混入头发和背景；
    2. 统计有效深度占比：手机/显示器屏幕几乎不反射红外光，该占比会很低；
    3. 用相机内参把像素反投影成三维点：X=(u-cx)*z/fx，Y=(v-cy)*z/fy，Z=z；
    4. 拟合最优平面并计算各点到平面的垂直距离，剔除乱深度后取
       “95% 分位 - 5% 分位”作为起伏值。

    真人脸是三维曲面（鼻尖凸出、下颌后收），起伏通常有 2~5 厘米；
    平面照片上的点近似共面，起伏只剩毫米级传感器噪声。
    先拟合平面可以抵消低头、抬头、照片倾斜造成的整体倾斜。

    无法判断时返回 (None, None, 有效深度占比)，例如面部区域几乎没有有效深度。
    """
    face_points = np.array(pixels, dtype=np.int32)
    mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(face_points), 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None, 0.0
    raw = depth_image[ys, xs].astype(np.float64) * depth_scale
    valid = raw > 0
    valid_ratio = float(valid.mean())
    # 有效深度占比过低 → 该区域不反射红外光（手机/显示器屏幕照片）。
    if valid_ratio < LIVENESS_MIN_VALID_RATIO:
        return None, None, valid_ratio
    ys, xs, depths = (
        ys[valid].astype(np.float64),
        xs[valid].astype(np.float64),
        raw[valid],
    )

    median_depth = float(np.median(depths))
    if depths.size < DEPTH_MIN_SAMPLES:
        return None, None, valid_ratio

    # 反投影成三维点云（课题要求的“点云”信息在这里用到）。
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(
        [(xs - cx) * depths / fx, (ys - cy) * depths / fy, depths]
    )

    # 反投影得到的点云拟合平面，起伏值即“脸部曲面偏离平面的程度”。
    relief = _plane_relief(points)
    return relief, median_depth, valid_ratio


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
    """头部姿态检测：先平滑俯仰角，再用迟滞区间 + 连续帧确认输出状态。

    俯仰角由面部点云法向量给出（数值连续、无镜像解跳变），这一层负责把它
    变成稳定的“正常/低头/抬头”状态：
    1. 中值滤波：吃掉单帧或两帧的异常角度；
    2. 指数平滑：抑制抖动，同时保留持续几十帧的真实头部动作；
    3. 迟滞区间：超过 HEAD_DOWN_ENTER_DEG 才算低头，低于
       -HEAD_UP_ENTER_DEG 才算抬头；回到 HEAD_DOWN_EXIT_DEG /
       -HEAD_UP_EXIT_DEG 以内才恢复为正常，避免角度在阈值附近来回跳；
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
        # 2) 指数平滑：让曲线变缓，但保留持续几十帧的真实头部动作。
        if self.smoothed is None:
            self.smoothed = filtered
        else:
            self.smoothed += HEAD_SMOOTH_ALPHA * (filtered - self.smoothed)
        if self.neutral is None:
            return "待校准（按 N 键）"

        delta = self.smoothed - self.neutral
        if delta > HEAD_DOWN_ENTER_DEG:
            target = "低头"
        elif delta < -HEAD_UP_ENTER_DEG:
            target = "抬头"
        elif self.state == "低头" and delta < HEAD_DOWN_EXIT_DEG:
            # 低头状态回到 +6° 以内就恢复正常；低于 -12° 已在上方转抬头。
            target = "正常"
        elif self.state == "抬头" and delta > -HEAD_UP_EXIT_DEG:
            # 抬头状态回到 -6° 以内就恢复正常；高于 +12° 已在上方转低头。
            target = "正常"
        elif self.state == "正常":
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


# ---------- 中文文字与下方信息面板 ----------
# OpenCV 自带字体不支持中文，面板统一用 Pillow 绘制后再转回 OpenCV 图像。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",    # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",  # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
)
_FONT_CACHE: dict[tuple[int, bool], ImageFont.ImageFont] = {}

PANEL_BG = (24, 27, 32)
PANEL_HEADER_BG = (34, 39, 46)
PANEL_LINE = (58, 66, 76)
PANEL_LABEL = (150, 160, 170)
PANEL_VALUE = (236, 240, 244)
PANEL_MUTED = (120, 130, 140)
PANEL_ACCENT = (80, 190, 255)
PANEL_GREEN = (82, 200, 120)
PANEL_AMBER = (240, 170, 60)
PANEL_RED = (235, 90, 90)
PANEL_OK_BG = (28, 56, 40)
PANEL_HEADER_HEIGHT = 36
PANEL_ROW_HEIGHT = 42
PANEL_STATUS_HEIGHT = 30
PANEL_HEIGHT = PANEL_HEADER_HEIGHT + 3 * PANEL_ROW_HEIGHT + PANEL_STATUS_HEIGHT
PANEL_REFRESH_SECONDS = 0.25


def _panel_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """按字号加载中文字体，字体对象只加载一次。"""
    key = (size, bold)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    candidates = (
        (_FONT_CANDIDATES[1],) + _FONT_CANDIDATES if bold else _FONT_CANDIDATES
    )
    font: ImageFont.ImageFont = ImageFont.load_default()
    for path in dict.fromkeys(candidates):
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            break
    _FONT_CACHE[key] = font
    return font


_PANEL_BASE_CACHE: dict[tuple[int, tuple[str, ...]], Image.Image] = {}


def _panel_base(width: int, labels: tuple[str, ...]) -> Image.Image:
    """绘制不随数值变化的表格底图并缓存。"""
    key = (width, labels)
    cached = _PANEL_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    base = Image.new("RGB", (width, PANEL_HEIGHT), PANEL_BG)
    draw = ImageDraw.Draw(base)
    title_font = _panel_font(19, bold=True)
    label_font = _panel_font(12)
    columns = 4
    column_width = width // columns

    # 顶栏：底色、左侧强调条和标题。
    draw.rectangle((0, 0, width - 1, PANEL_HEADER_HEIGHT - 1), fill=PANEL_HEADER_BG)
    draw.rectangle((0, 0, 3, PANEL_HEADER_HEIGHT - 1), fill=PANEL_ACCENT)
    draw.text(
        (16, 7), "课题 A · D455 RGB-D 人体状态检测",
        font=title_font, fill=PANEL_VALUE,
    )

    # 表格分隔线。
    for column in range(1, columns):
        x = column * column_width
        draw.line(
            (x, PANEL_HEADER_HEIGHT, x, PANEL_HEIGHT - PANEL_STATUS_HEIGHT),
            fill=PANEL_LINE,
        )
    for row in range(1, 3):
        y = PANEL_HEADER_HEIGHT + row * PANEL_ROW_HEIGHT
        draw.line((0, y, width - 1, y), fill=PANEL_LINE)

    # 静态标签。
    for index, label in enumerate(labels[: columns * 3]):
        row, column = divmod(index, columns)
        x = column * column_width + 14
        y = PANEL_HEADER_HEIGHT + row * PANEL_ROW_HEIGHT
        draw.text((x, y + 4), label, font=label_font, fill=PANEL_LABEL)

    _PANEL_BASE_CACHE[key] = base
    return base


def build_info_panel(
    width: int,
    cells: list[tuple[str, str, tuple[int, int, int]]],
    timestamp: str,
    status_text: str,
    status_bg: tuple[int, int, int],
    status_fg: tuple[int, int, int],
) -> np.ndarray:
    """把检测结果排成视频下方的信息表，避免文字遮挡画面。

    cells 固定为 4 列 × 3 行，每项是（标签, 数值, RGB 颜色）。
    标签、分隔线和标题缓存在底图中，每帧只重绘数值、时间戳和状态条。
    """
    labels = tuple(label for label, _, _ in cells)
    panel = _panel_base(width, labels).copy()
    draw = ImageDraw.Draw(panel)
    value_font = _panel_font(16)
    status_font = _panel_font(15, bold=True)
    small_font = _panel_font(13)
    columns = 4
    column_width = width // columns

    # 顶栏动态内容：时间戳和运行状态圆点。
    timestamp_box = draw.textbbox((0, 0), timestamp, font=small_font)
    timestamp_width = timestamp_box[2] - timestamp_box[0]
    timestamp_x = width - 16 - timestamp_width
    draw.text((timestamp_x, 10), timestamp, font=small_font, fill=PANEL_MUTED)
    dot_color = (
        PANEL_RED if status_bg == PANEL_RED
        else PANEL_AMBER if status_bg == PANEL_AMBER
        else PANEL_GREEN
    )
    draw.ellipse((timestamp_x - 20, 13, timestamp_x - 10, 23), fill=dot_color)

    # 动态数值。
    for index, (_, value, color) in enumerate(cells[: columns * 3]):
        row, column = divmod(index, columns)
        x = column * column_width + 14
        y = PANEL_HEADER_HEIGHT + row * PANEL_ROW_HEIGHT
        draw.text((x, y + 18), value, font=value_font, fill=color)

    # 底部状态条：正常、遮挡或报警三种状态。
    status_y = PANEL_HEIGHT - PANEL_STATUS_HEIGHT
    draw.rectangle((0, status_y, width - 1, PANEL_HEIGHT - 1), fill=status_bg)
    draw.text((16, status_y + 6), status_text, font=status_font, fill=status_fg)
    draw.ellipse(
        (width - 22, status_y + 10, width - 12, status_y + 20),
        fill=status_fg,
    )

    return cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)


def face_bounds(pixels: list[np.ndarray]) -> tuple[int, int, int, int, int]:
    """返回人脸框和面积，面积用于在多人画面中选择主目标。"""
    xs = [int(point[0]) for point in pixels]
    ys = [int(point[1]) for point in pixels]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2, y2, max(0, x2 - x1) * max(0, y2 - y1)


def select_main_face(
    faces, width: int, height: int
) -> Optional[tuple[object, list[np.ndarray], tuple[int, int, int, int, int]]]:
    """把检测到的多张脸转换为像素坐标，返回面积最大的主目标。"""
    candidates = []
    for face in faces:
        pixels = landmark_pixels(face, width, height)
        candidates.append((face, pixels, face_bounds(pixels)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2][4])


def eye_occlusion_hint(color_image: np.ndarray, pixels: list[np.ndarray]) -> bool:
    """判断双眼是否同时呈现“很暗且纹理很少”的遮挡特征。

    这是给现场演示用的保守提示，不是严格的眼镜分类器。正常眼镜通常不会
    同时满足两个条件；墨镜或大面积遮挡时，程序暂停眨眼计数，避免输出假数据。
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    def roi(indices: tuple[int, ...], padding: int = 5) -> np.ndarray:
        points = np.array([pixels[index] for index in indices], dtype=np.float32)
        x1 = max(0, int(points[:, 0].min()) - padding)
        x2 = min(color_image.shape[1], int(points[:, 0].max()) + padding + 1)
        y1 = max(0, int(points[:, 1].min()) - padding)
        y2 = min(color_image.shape[0], int(points[:, 1].max()) + padding + 1)
        return gray[y1:y2, x1:x2]

    face_roi = roi((1, 152, 33, 263, 61, 291), padding=0)
    if face_roi.size == 0:
        return False
    face_mean = float(face_roi.mean())
    occluded = []
    for indices in (
        (33, 160, 158, 133, 153, 144),
        (263, 387, 385, 362, 380, 373),
    ):
        eye_roi = roi(indices)
        if eye_roi.size < 12:
            return False
        eye_mean = float(eye_roi.mean())
        eye_std = float(eye_roi.std())
        occluded.append(
            eye_mean < OCCLUSION_DARK_MEAN
            and eye_mean < face_mean * OCCLUSION_DARK_RATIO
            and eye_std < OCCLUSION_MAX_STD
        )
    return all(occluded)


def save_rgbd_snapshot(
    output_dir: Path,
    color_image: np.ndarray,
    depth_image: np.ndarray,
    display: np.ndarray,
    tag: str = "",
) -> str:
    """保存彩色、原始深度和拼接预览图，返回本次文件名前缀。"""
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    prefix = f"{stamp}_{tag}" if tag else stamp
    cv2.imwrite(str(output_dir / f"{prefix}_color.png"), color_image)
    cv2.imwrite(str(output_dir / f"{prefix}_depth.png"), depth_image)
    cv2.imwrite(str(output_dir / f"{prefix}_preview.png"), display)
    return prefix


EVENT_FIELDS = (
    "timestamp",
    "event",
    "face_count",
    "main_face_index",
    "liveness",
    "head",
    "pitch",
    "ear",
    "relief",
    "distance",
    "valid_ratio",
    "blink_count",
    "alarm",
)


class EventLogger:
    """把关键状态写成 CSV，便于重复实验和报告统计。"""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "events.csv"

    def log(self, event: str, **values) -> None:
        self.path.parent.mkdir(exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        row = {field: values.get(field, "") for field in EVENT_FIELDS}
        row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
        row["event"] = event
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


class ResultRecorder:
    """按 R 键录制带检测结果的 RGB-D 拼接画面。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def toggle(self, frame: np.ndarray) -> bool:
        if self.active:
            self.stop()
            return False
        self.output_dir.mkdir(exist_ok=True)
        path = self.output_dir / f"result_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (frame.shape[1], frame.shape[0]),
        )
        if not writer.isOpened():
            writer.release()
            print(f"无法创建录制文件：{path}")
            return False
        self.writer = writer
        self.path = path
        print(f"开始录制：{path}")
        return True

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def stop(self) -> None:
        if self.writer is None:
            return
        path = self.path
        self.writer.release()
        self.writer = None
        self.path = None
        print(f"已停止录制：{path}")


class SessionStats:
    """累计现场演示需要的帧率、检测和状态切换统计。"""

    def __init__(self) -> None:
        self.fps_samples = deque(maxlen=STATS_WINDOW)
        self.total_frames = 0
        self.face_frames = 0
        self.head_switches = 0
        self.last_head_status: Optional[str] = None
        self.alarm_count = 0

    def update(self, fps: float, face_detected: bool, head_status: str) -> None:
        self.total_frames += 1
        if fps > 0:
            self.fps_samples.append(fps)
        if face_detected:
            self.face_frames += 1
        if head_status in ("正常", "低头", "抬头"):
            if self.last_head_status is not None and head_status != self.last_head_status:
                self.head_switches += 1
            self.last_head_status = head_status

    @property
    def face_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.face_frames / self.total_frames

    def fps_text(self) -> str:
        if not self.fps_samples:
            return "--"
        values = np.asarray(self.fps_samples, dtype=np.float64)
        return f"{np.percentile(values, 50):.1f}/{np.percentile(values, 95):.1f}"

    def summary_text(self, blink_count: int) -> str:
        return (
            f"帧数={self.total_frames}, 检测到人脸={self.face_frames}, "
            f"人脸帧占比={self.face_ratio:.1%}, 头部切换={self.head_switches}, "
            f"报警={self.alarm_count}, 眨眼={blink_count}, "
            f"FPS p50/p95={self.fps_text()}"
        )


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
    logger = EventLogger(output_dir)
    recorder = ResultRecorder(output_dir)
    stats = SessionStats()

    # 以下变量是跨帧状态：眨眼计数、正常头部姿态、活体判断稳定计数等。
    previous_timestamp_ms = None
    blink = BlinkDetector()
    head = HeadPoseDetector()
    # 活体结果只取“真人/照片”两种；未检测到人脸时为 None（界面显示 --）。
    liveness_label: Optional[str] = None
    real_counter = 0
    photo_counter = 0
    face_count = 0
    main_face_index = -1
    valid_ratio = 0.0
    eye_occlusion_counter = 0
    eye_occluded = False
    alarm_since: Optional[float] = None
    alarm_active = False
    head_status = "--"
    ear_value = 0.0
    depth_relief: Optional[float] = None
    face_distance: Optional[float] = None

    # 信息面板只在数值刷新周期或状态切换时重建，避免每帧重绘文字。
    last_panel: Optional[np.ndarray] = None
    last_panel_time = 0.0
    last_panel_signature: Optional[tuple] = None

    def log_state(event: str, alarm_value: bool) -> None:
        """记录当前帧的关键状态，避免每种事件重复展开字段。"""
        logger.log(
            event,
            face_count=face_count,
            main_face_index=main_face_index + 1,
            liveness=liveness_label,
            head=head_status,
            pitch=head.raw,
            ear=ear_value,
            relief=depth_relief,
            distance=face_distance,
            valid_ratio=valid_ratio,
            blink_count=blink.count,
            alarm=alarm_value,
        )

    print(f"已启动 D455，深度比例：{depth_scale:.6f} 米/单位")
    print("快捷键：Q/Esc 退出 | S 保存截图 | R 开始/停止录制 | N 校准正常头部姿态")

    try:
        # FaceMesh 对每一帧图像输出人脸关键点。
        # refine_landmarks=True 会提供更细致的眼睛和嘴部关键点。
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=MAX_FACES, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                # 每帧记录一次单调时间，供眨眼冷却和低头报警按真实时间使用。
                frame_time = time.monotonic()
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
                head_status = "--"
                ear_value = 0.0
                pitch_value = None
                depth_relief: Optional[float] = None
                face_distance: Optional[float] = None

                faces = result.multi_face_landmarks or []
                face_count = len(faces)
                selected = select_main_face(
                    faces, color_image.shape[1], color_image.shape[0]
                )
                if selected is not None:
                    face_landmarks, pixels, bounds = selected
                    eye_hint = eye_occlusion_hint(color_image, pixels)
                    main_face_index = next(
                        (
                            index
                            for index, face in enumerate(faces)
                            if face is face_landmarks
                        ),
                        -1,
                    )
                    # 其余人脸只画框，不参与眨眼和低头判定，避免目标来回切换。
                    for index, face in enumerate(faces):
                        if face is face_landmarks:
                            continue
                        other_pixels = landmark_pixels(
                            face, color_image.shape[1], color_image.shape[0]
                        )
                        x1, y1, x2, y2, _ = face_bounds(other_pixels)
                        cv2.rectangle(
                            color_image, (x1, y1), (x2, y2), (160, 160, 160), 1
                        )
                        cv2.putText(
                            color_image,
                            f"face {index + 1}",
                            (x1, max(16, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (160, 160, 160),
                            1,
                            cv2.LINE_AA,
                        )
                    # 只画脸轮廓、双眼和嘴唇：完整三角网格每帧要画 2556 条线，
                    # 实测占 14~20 ms/帧；换成 124 条轮廓线后约 2.2 ms/帧，
                    # 为 FaceMesh 和眨眼判定留出时间，也更容易观察眼睛开合。
                    mp_drawing.draw_landmarks(
                        color_image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 200, 255), thickness=1, circle_radius=1
                        ),
                    )
                    x1, y1, x2, y2, _ = bounds
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 180, 0), 2)

                    # 左右眼分别计算 EAR；眼部遮挡时暂停眨眼计数。
                    left_ear = eye_aspect_ratio(pixels, (33, 160, 158, 133, 153, 144))
                    right_ear = eye_aspect_ratio(pixels, (263, 387, 385, 362, 380, 373))
                    ear_value = (left_ear + right_ear) / 2.0
                    if eye_hint:
                        eye_occlusion_counter = min(
                            eye_occlusion_counter + 1, OCCLUSION_HOLD_FRAMES
                        )
                    else:
                        eye_occlusion_counter = max(eye_occlusion_counter - 1, 0)
                    eye_occluded = eye_occlusion_counter >= OCCLUSION_HOLD_FRAMES
                    if eye_occluded:
                        blink.reset()
                    else:
                        blink.update(left_ear, right_ear, frame_time)

                    # 用面部点云法向量估计俯仰角，再交给 HeadPoseDetector 判定。
                    pitch_value = face_pitch_from_depth(
                        depth_image, pixels, depth_scale, camera_matrix
                    )
                    head_status = head.update(pitch_value)

                    # 真人脸是三维曲面，平面照片是平的，屏幕照片取不到有效深度。
                    depth_relief, face_distance, valid_ratio = face_depth_relief(
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
                else:
                    main_face_index = -1
                    valid_ratio = 0.0
                    eye_occlusion_counter = 0
                    eye_occluded = False
                    blink.reset()
                    head.reset()
                    real_counter = 0
                    photo_counter = 0
                    liveness_label = None

                stats.update(fps, face_count > 0, head_status)
                alarm_started = False
                now_monotonic = frame_time
                if head_status == "低头" and liveness_label == "真人":
                    if alarm_since is None:
                        alarm_since = now_monotonic
                    if (
                        not alarm_active
                        and now_monotonic - alarm_since >= ALARM_HOLD_SECONDS
                    ):
                        alarm_active = True
                        alarm_started = True
                        stats.alarm_count += 1
                        log_state("alarm_start", True)
                else:
                    if alarm_active:
                        log_state("alarm_end", False)
                    alarm_active = False
                    alarm_since = None

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
                head_text = head_status.replace("（按 N 键）", "")
                if face_count > 0:
                    face_text = f"{face_count} 张 · 主目标 {main_face_index + 1}"
                    face_color = PANEL_VALUE
                else:
                    face_text = "未检测"
                    face_color = PANEL_MUTED
                if face_count == 0:
                    blink_text = f"{blink.count} 次 · 当前未检测"
                    blink_color = PANEL_MUTED
                elif eye_occluded:
                    blink_text = "计数暂停 · 眼部遮挡"
                    blink_color = PANEL_AMBER
                else:
                    blink_text = (
                        f"{blink.count} 次 · EAR {ear_value:.2f}"
                        f" / 基线 {blink.baseline:.2f}"
                    )
                    blink_color = PANEL_VALUE

                if alarm_active and alarm_since is not None:
                    status_key = "alarm"
                    status_text = f"低头报警 · 已持续 {frame_time - alarm_since:.1f} 秒"
                    status_bg, status_fg = PANEL_RED, (255, 255, 255)
                elif eye_occluded:
                    status_key = "occlusion"
                    status_text = "眼部遮挡 · 眨眼计数暂停"
                    status_bg, status_fg = PANEL_AMBER, (24, 27, 32)
                else:
                    status_key = "normal"
                    status_text = "系统运行中 · 检测正常"
                    status_bg, status_fg = PANEL_OK_BG, PANEL_GREEN

                cells = [
                    (
                        "活体判定",
                        liveness_text,
                        PANEL_GREEN if liveness_label == "真人"
                        else PANEL_AMBER if liveness_label == "照片"
                        else PANEL_MUTED,
                    ),
                    (
                        "头部状态",
                        head_text,
                        PANEL_RED if head_status == "低头"
                        else PANEL_ACCENT if head_status == "抬头"
                        else PANEL_GREEN if head_status == "正常"
                        else PANEL_MUTED,
                    ),
                    ("眨眼计数", blink_text, blink_color),
                    (
                        "低头报警",
                        "已触发" if alarm_active else "正常",
                        PANEL_RED if alarm_active else PANEL_GREEN,
                    ),
                    (
                        "俯仰角",
                        pitch_text,
                        PANEL_RED if head_status == "低头"
                        else PANEL_ACCENT if head_status == "抬头"
                        else PANEL_VALUE,
                    ),
                    (
                        "面部起伏",
                        relief_text,
                        PANEL_MUTED if depth_relief is None else PANEL_VALUE,
                    ),
                    (
                        "深度有效",
                        f"{valid_ratio:.1%}",
                        PANEL_AMBER
                        if valid_ratio < LIVENESS_MIN_VALID_RATIO
                        else PANEL_VALUE,
                    ),
                    (
                        "中心距离",
                        f"{center_depth_m:.2f} 米",
                        PANEL_AMBER
                        if center_depth_m < LIVENESS_MIN_RELIABLE_DEPTH_M
                        else PANEL_VALUE,
                    ),
                    ("人脸目标", face_text, face_color),
                    (
                        "处理帧率",
                        f"{fps:.1f} FPS · p50/p95 {stats.fps_text()}",
                        PANEL_AMBER if fps < 8.0 else PANEL_VALUE,
                    ),
                    (
                        "结果录制",
                        "开" if recorder.active else "关",
                        PANEL_RED if recorder.active else PANEL_MUTED,
                    ),
                    (
                        "会话统计",
                        f"头部切换 {stats.head_switches} · 报警 {stats.alarm_count}",
                        PANEL_VALUE,
                    ),
                ]
                # 上方保留彩色/深度两路画面，下方嵌入 4 列 × 3 行状态表。
                video = np.hstack((color_image, depth_color))
                panel_signature = (
                    status_key,
                    liveness_label,
                    head_status,
                    recorder.active,
                    face_count,
                    main_face_index,
                )
                if (
                    last_panel is None
                    or frame_time - last_panel_time >= PANEL_REFRESH_SECONDS
                    or panel_signature != last_panel_signature
                ):
                    last_panel = build_info_panel(
                        video.shape[1],
                        cells,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        status_text=status_text,
                        status_bg=status_bg,
                        status_fg=status_fg,
                    )
                    last_panel_time = frame_time
                    last_panel_signature = panel_signature
                display = np.vstack((video, last_panel))
                if alarm_started:
                    snapshot = save_rgbd_snapshot(
                        output_dir, color_image, depth_image, display, "alarm"
                    )
                    print(f"已保存低头报警证据：{snapshot}")
                recorder.write(display)
                # 窗口标题用 ASCII：OpenCV 的 HighGUI 标题栏显示中文会乱码。
                cv2.imshow("Project A - D455 RGB-D", display)

                # waitKey 同时负责刷新窗口和读取键盘输入。
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    # 用户保持正常正视姿态按 N，把平滑后的当前角度作为基准。
                    neutral = head.calibrate()
                    if neutral is not None:
                        log_state("calibrate", alarm_active)
                        print(f"已记录正常头部俯仰角：{neutral:.2f} 度")
                elif key == ord("s"):
                    # 保存原始彩色图、原始深度图和便于答辩展示的拼接预览图。
                    stamp = save_rgbd_snapshot(
                        output_dir, color_image, depth_image, display
                    )
                    log_state("snapshot", alarm_active)
                    print(f"已保存 RGB-D 数据：{stamp}")
                elif key == ord("r"):
                    recording = recorder.toggle(display)
                    if recording:
                        recorder.write(display)
                    log_state(
                        "record_start" if recording else "record_stop",
                        alarm_active,
                    )
                elif key in (ord("q"), 27):
                    # Q 或 Esc 退出主循环。
                    break
    finally:
        # 无论正常退出还是运行中发生异常，都要释放相机和 OpenCV 窗口。
        recorder.stop()
        if alarm_active:
            log_state("alarm_end", False)
        log_state("session_end", False)
        summary = stats.summary_text(blink.count)
        output_dir.mkdir(exist_ok=True)
        (output_dir / "session_summary.txt").write_text(
            f"结束时间：{datetime.now().isoformat(timespec='seconds')}\n{summary}\n",
            encoding="utf-8",
        )
        print(f"会话统计：{summary}")
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
