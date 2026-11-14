from pathlib import Path
import cv2
from ultralytics import YOLO

# Change this ratio if the counting line should be elsewhere in the video.
LINE_POSITION_RATIO = 0.50
CONFIDENCE = 0.35

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
line_x = int(width * LINE_POSITION_RATIO)

Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/entry_exit_count.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
model = YOLO("yolov8n.pt")
previous_x = {}
counted_ids = set()
entering = 0
leaving = 0

print("Counting vehicles crossing the vertical line. Press Q to stop.")
print("Assumption: left-to-right = entering; right-to-left = leaving.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=CONFIDENCE, verbose=False)[0]
    annotated = frame.copy()
    cv2.line(annotated, (line_x, 0), (line_x, height), (0, 255, 255), 2)

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            old_x = previous_x.get(track_id)
            if old_x is not None and track_id not in counted_ids:
                if old_x < line_x <= center_x:
                    entering += 1
                    counted_ids.add(track_id)
                elif old_x > line_x >= center_x:
                    leaving += 1
                    counted_ids.add(track_id)
            previous_x[track_id] = center_x
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 220, 0), 2)
            cv2.putText(annotated, f"ID {track_id}", (int(x1), max(20, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    cv2.putText(annotated, f"Entering: {entering}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Leaving: {leaving}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    writer.write(annotated)
    cv2.imshow("VisionPort - Entry and Exit Count", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
writer.release()
cv2.destroyAllWindows()
print("Entry count:", entering)
print("Exit count:", leaving)
print("Output saved: outputs/entry_exit_count.mp4")
