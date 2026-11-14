# Task 06 – Car Detection Using YOLO

**Objective:** Detect cars in a traffic video using YOLO and draw bounding boxes.

**File:** `detect_cars_yolo.py`

**Run:**

```powershell
py -m pip install -r ..\requirements.txt
py detect_cars_yolo.py
```

Enter the traffic video path. YOLO class ID `2` represents a car in the COCO model. The annotated result is saved under `outputs/car_detection/`.

**Expected output:** A video preview with car bounding boxes and a saved annotated video.

**Note:** The first run may download `yolov8n.pt` automatically.
