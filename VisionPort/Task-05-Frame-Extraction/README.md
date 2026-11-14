# Task 05 – Frame Extraction

**Objective:** Extract still images (frames) from a traffic video.

**File:** `extract_video_frames.py`

**Run:**

```powershell
py extract_video_frames.py
```

Enter a video path. The program saves every 30th frame under `extracted_frames/<video-name>/`. To save every frame, change `FRAME_INTERVAL = 30` to `FRAME_INTERVAL = 1` at the top of the code.

**Expected output:** JPG images named `frame_000000.jpg`, `frame_000030.jpg`, and so on.
