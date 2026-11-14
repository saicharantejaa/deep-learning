from pathlib import Path
from ultralytics import YOLO

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")

# COCO IDs: motorcycle = 3, bus = 5, truck = 7
model.predict(
    source=video_path,
    classes=[3, 5, 7],
    conf=0.35,
    show=True,
    save=True,
    project="outputs",
    name="multiple_vehicle_detection",
    exist_ok=True
)

print("Detection completed. Check outputs/multiple_vehicle_detection.")
