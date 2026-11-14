# Task 13 – Approximate Vehicle Speed

**Objective:** Estimate a vehicle's approximate speed from its tracked movement across video frames.

**File:** `estimate_vehicle_speed.py`

**Run:**

```powershell
py estimate_vehicle_speed.py
```

Enter the traffic video path, then enter a **measured metres-per-pixel value** for that fixed camera view. Example: if a known 5 metre distance spans 100 pixels, enter `0.05`.

The script uses YOLO + ByteTrack, calculates centre movement between frames, and estimates km/h. The annotated video is saved as `outputs/speed_estimation.mp4`.

**Important limitation:** This is only an approximate estimate. A valid camera calibration, fixed camera, known road distance, suitable frame rate and stable tracking are required. An arbitrary calibration value will produce meaningless speeds. Perspective also means a single metres-per-pixel value may not be accurate across the whole image.
