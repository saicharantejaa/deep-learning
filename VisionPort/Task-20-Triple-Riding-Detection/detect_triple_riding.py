from pathlib import Path
import cv2
from ultralytics import YOLO

CONFIDENCE = 0.35
# Extend the motorcycle box upwards to include riders above the vehicle body.
RIDER_AREA_HEIGHT_MULTIPLIER = 1.4
RIDER_AREA_WIDTH_MULTIPLIER = 0.25

video_path = input("Enter traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error: Could not open video.")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS) or 25
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/triple_riding_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
model = YOLO("yolov8n.pt")
frame_number = 0
triple_riding_frames = 0

print("Detecting motorcycles and people around them. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model(frame, classes=[0, 3], conf=CONFIDENCE, verbose=False)[0]
    annotated = frame.copy()
    person_boxes = []
    motorcycle_boxes = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            coordinates = box.xyxy[0].cpu().tolist()
            if class_id == 0:
                person_boxes.append(coordinates)
            elif class_id == 3:
                motorcycle_boxes.append(coordinates)

    frame_alert = False
    for bike in motorcycle_boxes:
        x1, y1, x2, y2 = map(int, bike)
        bike_width = x2 - x1
        bike_height = y2 - y1
        # Count person-box centres in an expanded region around the motorcycle.
        rx1 = max(0, int(x1 - bike_width * RIDER_AREA_WIDTH_MULTIPLIER))
        rx2 = min(width, int(x2 + bike_width * RIDER_AREA_WIDTH_MULTIPLIER))
        ry1 = max(0, int(y1 - bike_height * RIDER_AREA_HEIGHT_MULTIPLIER))
        ry2 = min(height, int(y2 + bike_height * 0.25))
        riders = 0
        for person in person_boxes:
            px1, py1, px2, py2 = person
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            if rx1 <= person_center[0] <= rx2 and ry1 <= person_center[1] <= ry2:
                riders += 1

        is_triple = riders > 2
        if is_triple:
            frame_alert = True
        color = (0, 0, 255) if is_triple else (0, 220, 0)
        label = f"TRIPLE RIDING: {riders}" if is_triple else f"Persons: {riders}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 1)
        cv2.putText(annotated, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if frame_alert:
        triple_riding_frames += 1
        cv2.putText(annotated, "TRIPLE RIDING ALERT", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    writer.write(annotated)
    cv2.imshow("VisionPort - Triple Riding Detection", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
    frame_number += 1

video.release()
writer.release()
cv2.destroyAllWindows()
print("Processed frames:", frame_number)
print("Frames with triple-riding alert:", triple_riding_frames)
print("Output saved: outputs/triple_riding_detection.mp4")
