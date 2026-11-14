# Task 09 – Vehicle Counting

**Objective:** Count cars, motorcycles, buses and trucks in one frame of a traffic video.

**File:** `count_vehicles.py`

**Run:**

```powershell
py count_vehicles.py
```

Enter a video path. The program reads the first frame, detects vehicle classes with YOLO, counts the detections and saves an annotated image as `outputs/vehicle_count.jpg`.

**Expected output:** The total vehicle count printed in PowerShell and on the saved image.

**Note:** This counts detections in one frame, not unique vehicles across the full video.
