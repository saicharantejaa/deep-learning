# Task 20 – Triple-Riding Detection

**Objective:** Flag a motorcycle when more than two person detections appear in an expanded region around it.

## Run

```powershell
py detect_triple_riding.py
```

Enter a traffic video path. Press `Q` to stop. Output: `outputs/triple_riding_detection.mp4`.

## Notes and limitations

The pretrained YOLOv8 COCO model detects the `person` and `motorcycle` classes. The script counts person-box centres inside an expanded motorcycle region (extended upward to include riders). It is a simple visual estimate, not a reliable legal determination. Occlusion, crowded scenes, pedestrians close to a motorcycle, camera angle, and missed person detections can cause false results. Use clear side-view footage and adjust `RIDER_AREA_HEIGHT_MULTIPLIER` / `RIDER_AREA_WIDTH_MULTIPLIER` for the camera.
