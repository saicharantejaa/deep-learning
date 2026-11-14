from collections import defaultdict, deque
from pathlib import Path
import cv2
from ultralytics import YOLO

# Set the expected direction for your camera view: RIGHT, LEFT, UP or DOWN.
EXPECTED_DIRECTION = "RIGHT"
MIN_MOVEMENT_PIXELS = 45
HISTORY_LENGTH = 12
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
Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/wrong_way_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
model = YOLO("yolov8n.pt")
paths = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
wrong_ids = set()

print("Expected direction:", EXPECTED_DIRECTION)
print("Checking vehicle movement. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break
    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=CONFIDENCE, verbose=False)[0]
    annotated = frame.copy()

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            paths[track_id].append(center)
            history = paths[track_id]
            if len(history) == HISTORY_LENGTH:
                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                wrong = ((EXPECTED_DIRECTION == "RIGHT" and dx < -MIN_MOVEMENT_PIXELS) or
                         (EXPECTED_DIRECTION == "LEFT" and dx > MIN_MOVEMENT_PIXELS) or
                         (EXPECTED_DIRECTION == "DOWN" and dy < -MIN_MOVEMENT_PIXELS) or
                         (EXPECTED_DIRECTION == "UP" and dy > MIN_MOVEMENT_PIXELS))
                if wrong:
                    wrong_ids.add(track_id)
            color = (0, 0, 255) if track_id in wrong_ids else (0, 220, 0)
            label = f"ID {track_id} WRONG WAY" if track_id in wrong_ids else f"ID {track_id}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, label, (int(x1), max(20, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, f"Wrong-way alerts: {len(wrong_ids)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    writer.write(annotated)
    cv2.imshow("VisionPort - Wrong Way Detection", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
writer.release()
cv2.destroyAllWindows()
print("Wrong-way vehicle IDs:", sorted(wrong_ids))
print("Output saved: outputs/wrong_way_detection.mp4")
