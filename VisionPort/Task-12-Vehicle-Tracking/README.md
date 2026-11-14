# Task 12 – Vehicle Tracking

**Objective:** Track cars, motorcycles, buses and trucks across video frames and display their tracking IDs.

**File:** `track_vehicles.py`

**Run:**

```powershell
py track_vehicles.py
```

Enter a traffic video path. YOLO tracking with the ByteTrack tracker assigns IDs to detections as they move through the video. The annotated video is saved as `outputs/vehicle_tracking.mp4`.

**Note:** IDs can change or be lost when vehicles are hidden, leave the frame, or are difficult to detect.
