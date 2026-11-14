from pathlib import Path
from ultralytics import YOLO

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")

print("Detecting cars. Close the preview window or press Q to stop if supported.")
model.predict(
    source=video_path,
    classes=[2],              # COCO class 2 = car
    conf=0.35,
    show=True,
    save=True,
    project="outputs",
    name="car_detection",
    exist_ok=True
)

print("Car detection completed. Check outputs/car_detection.")
