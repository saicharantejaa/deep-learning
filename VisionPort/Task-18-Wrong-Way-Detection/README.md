# Task 18 – Wrong-Way Vehicle Detection

**Objective:** Alert when a tracked vehicle moves opposite to the configured normal traffic direction.

## Run

```powershell
py detect_wrong_way.py
```

Enter a traffic video path. Press `Q` to stop. Output: `outputs/wrong_way_detection.mp4`.

## Configuration

Set `EXPECTED_DIRECTION` to `RIGHT`, `LEFT`, `UP` or `DOWN`, according to the normal travel direction in the camera image. Adjust `MIN_MOVEMENT_PIXELS` if the camera resolution or viewing distance is different.

This simple demonstration assumes the road has one expected direction in the viewed area. It is not suitable for a multi-lane, two-way junction without adding lane-specific zones and direction rules. Tracking errors and camera movement can cause false alerts.
