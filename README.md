# Project A: D455 RGB-D Preview

## Run

Use the local `realsense_a` Conda interpreter in PyCharm, connect the D455 to a USB 3 port, then run:

```powershell
python preview_rgbd.py
```

The left panel is the color image. The right panel is the aligned pseudo-color depth image. The green dot marks the center pixel and displays its distance in meters. The frame rate is shown in the upper-left corner.

Press `S` to save the current color image, raw 16-bit depth image, and side-by-side preview in `captures/`. Press `Q` or `Esc` to close the preview.

## Before Running

Close Intel RealSense Viewer before running this program, because only one program can use the camera at a time.
