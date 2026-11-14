from pathlib import Path
from ultralytics import YOLO

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

model = YOLO("yolov8n.pt")
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

results = model.predict(
    source=video_path,
    classes=vehicle_ids,
    conf=0.25,
    show=True,
    save=True,
    project="outputs",
    name="confidence_scores",
    exist_ok=True
)

# Print each detected vehicle name and its confidence value.
for frame_number, result in enumerate(results, start=1):
    if result.boxes is None:
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        vehicle_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"Frame {frame_number}: {vehicle_name} - Confidence: {confidence:.2f}")

print("Completed. YOLO also displays labels and confidence on the preview.")
print("Saved results in outputs/confidence_scores.")
