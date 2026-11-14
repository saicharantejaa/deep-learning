# Task 07 – Detect Bikes, Buses and Trucks

**Objective:** Detect motorcycles, buses and trucks in a traffic video using YOLO.

**File:** `detect_multiple_vehicles.py`

**Run:**

```powershell
py detect_multiple_vehicles.py
```

Enter a traffic video path. The program filters YOLO's COCO classes: motorcycle (`3`), bus (`5`) and truck (`7`). Results are saved under `outputs/multiple_vehicle_detection/`.

**Expected output:** Video preview and saved video with bounding boxes around detected motorcycles, buses and trucks.
