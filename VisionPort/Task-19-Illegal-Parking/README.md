# Task 19 – Illegal Parking Detection

**Objective:** Detect vehicles that remain stationary inside a configured restricted polygon for a specified duration.

## Run

```powershell
py detect_illegal_parking.py
```

Enter a traffic video path. Press `Q` to stop. Output: `outputs/illegal_parking_detection.mp4`.

## Configure the restricted area

Edit `RESTRICTED_ZONE` near the top of the script. Its four polygon points are ratios from 0.0 to 1.0, measured from the top-left of the video frame. The default polygon is only an example and will probably need adjustment for your footage. Set `PARKED_SECONDS` to the required waiting threshold.

This is an approximate stationary-in-zone demonstration. A slow-moving vehicle may be mistaken for a parked vehicle, and tracking loss may reset or misattribute timing. Use a fixed camera and validate the zone before presenting results.
