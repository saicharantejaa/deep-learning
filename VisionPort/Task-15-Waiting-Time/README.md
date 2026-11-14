# Task 15 – Traffic Waiting-Time Estimate

**Objective:** Estimate how long each tracked vehicle remains almost stationary in the video.

**File:** `estimate_waiting_time.py`

**Run:**

```powershell
py estimate_waiting_time.py
```

Enter a traffic-junction video path. YOLO + ByteTrack assigns vehicle IDs. If a vehicle's centre moves no more than 5 pixels between consecutive frames, its stationary counter increases. The program displays the time beside the vehicle and saves:

- `outputs/waiting_time_video.mp4`
- `outputs/waiting_time_report.csv`

**Important limitation:** This estimates continuous stationary time, not a true signal-cycle or queue waiting-time measurement. A fixed camera, good tracking and suitable footage are necessary. Adjust `STATIONARY_PIXELS` for the video resolution and camera.
