# Task 14 – Traffic Signal Status Detection

**Objective:** Detect traffic-light regions and estimate whether the visible light is red, yellow or green.

**File:** `detect_traffic_signal.py`

**Run:**

```powershell
py detect_traffic_signal.py
```

Enter a traffic video path. YOLO detects traffic-light regions (COCO class `9`); OpenCV checks the colours inside each detected region. The annotated video is saved as `outputs/traffic_signal_status.mp4`.

**Important limitation:** This is an approximate colour-based demonstration. Small, distant, blurry, shaded or partially hidden lights may be missed or classified as `UNKNOWN`. Use clear traffic-light footage for testing.
