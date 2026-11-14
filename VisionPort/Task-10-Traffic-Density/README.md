# Task 10 – Traffic Density

**Objective:** Classify traffic as low, medium or high based on the number of vehicles detected in a video frame.

**File:** `calculate_traffic_density.py`

**Run:**

```powershell
py calculate_traffic_density.py
```

Enter a video path. The script analyses the first frame and applies these demonstration thresholds:

- 0–5 vehicles: LOW
- 6–15 vehicles: MEDIUM
- More than 15 vehicles: HIGH

The annotated image is saved as `outputs/traffic_density.jpg`.

**Note:** These are simple example thresholds, not universal traffic-engineering standards. Tune them for the camera view and road area.
