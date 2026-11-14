# Task 11 – Congestion Detection

**Objective:** Show a congestion alert when many vehicles remain in a selected junction area for a continuous period.

**File:** `detect_congestion.py`

**Run:**

```powershell
py detect_congestion.py
```

Enter a traffic-junction video path. The script uses a central/lower rectangular region as the junction area. It displays an alert when at least 8 detected vehicles remain in that region for 3 seconds. The annotated result is saved as `outputs/congestion_detection.mp4`.

**Important limitation:** This is a simple demonstration using vehicle count and time, not a full traffic-engineering congestion model. Adjust the region and threshold for your video. A fixed camera and a clear junction view are recommended.
