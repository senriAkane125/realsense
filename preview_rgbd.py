"""Intel RealSense D455 RGB-D preview for project A."""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as error:
        print("Could not start the RealSense camera.")
        print("Connect the D455 to a USB 3 port and close RealSense Viewer first.")
        print(f"Details: {error}")
        sys.exit(1)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    output_dir = Path("captures")
    previous_timestamp_ms = None
    print(f"D455 started. Depth scale: {depth_scale:.6f} m/unit")
    print("Press S to save an RGB-D capture. Press Q or Esc to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            center_x = depth_frame.get_width() // 2
            center_y = depth_frame.get_height() // 2
            distance_m = depth_frame.get_distance(center_x, center_y)
            timestamp_ms = color_frame.get_timestamp()
            fps = 0.0
            if previous_timestamp_ms is not None and timestamp_ms > previous_timestamp_ms:
                fps = 1000.0 / (timestamp_ms - previous_timestamp_ms)
            previous_timestamp_ms = timestamp_ms

            cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(
                color_image,
                f"Center depth: {distance_m:.2f} m",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color_image,
                f"FPS: {fps:.1f}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            display = np.hstack((color_image, depth_color))
            cv2.imshow("Project A - D455 RGB-D Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                output_dir.mkdir(exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                color_path = output_dir / f"{stamp}_color.png"
                depth_path = output_dir / f"{stamp}_depth.png"
                preview_path = output_dir / f"{stamp}_preview.png"
                cv2.imwrite(str(color_path), color_image)
                cv2.imwrite(str(depth_path), depth_image)
                cv2.imwrite(str(preview_path), display)
                print(f"Saved: {color_path}, {depth_path}, {preview_path}")
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
