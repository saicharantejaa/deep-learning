from collections import defaultdict, deque
from pathlib import Path
import math
import cv2
from ultralytics import YOLO

# The default intersection area is the centre half of the frame.
# Change these ratios to fit the actual junction in your video.
ROI = (0.25, 0.25, 0.75, 0.75)  # left, top, right, bottom
ALLOWED_TURNS = {"LEFT", "STRAIGHT"}  # Configure for the selected incoming lane.
CONFIDENCE = 0.35


def classify_turn(points):
    if len(points) < 8:
        return "UNKNOWN"
    start = points[0]
    middle = points[len(points) // 2]
    end = points[-1]
    first = (middle[0] - start[0], middle[1] - start[1])
    last = (end[0] - middle[0], end[1] - middle[1])
    if math.hypot(*first) < 10 or math.hypot(*last) < 10:
        return "UNKNOWN"
    dot = first[0] * last[0] + first[1] * last[1]
    cross = first[0] * last[1] - first[1] * last[0]
    angle = math.degrees(math.atan2(cross, dot))
    if abs(angle) < 30:
        return "STRAIGHT"
    if abs(angle) > 150:
        return "U-TURN"
    # Image y increases downwards; this is screen-relative left/right.
    return "RIGHT" if angle > 0 else "LEFT"


video_path = input("Enter intersection video path: ").strip().strip('"')
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
roi = (int(width * ROI[0]), int(height * ROI[1]),
       int(width * ROI[2]), int(height * ROI[3]))
Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/wrong_turn_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
model = YOLO("yolov8n.pt")
paths = defaultdict(lambda: deque(maxlen=24))
was_inside = set()
finished = set()
wrong_ids = set()

print("Analysing vehicle paths through the intersection. Press Q to stop.")
print("Allowed turns:", ", ".join(sorted(ALLOWED_TURNS)))

while True:
    success, frame = video.read()
    if not success:
        break
    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=CONFIDENCE, verbose=False)[0]
    annotated = frame.copy()
    x1r, y1r, x2r, y2r = roi
    cv2.rectangle(annotated, (x1r, y1r), (x2r, y2r), (255, 180, 0), 2)

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            paths[track_id].append(center)
            inside = x1r <= center[0] <= x2r and y1r <= center[1] <= y2r
            if inside:
                was_inside.add(track_id)
            if track_id in was_inside and not inside and track_id not in finished:
                turn = classify_turn(list(paths[track_id]))
                finished.add(track_id)
                if turn != "UNKNOWN" and turn not in ALLOWED_TURNS:
                    wrong_ids.add(track_id)
                    print(f"ALERT: Vehicle ID {track_id} made a {turn} turn")
            color = (0, 0, 255) if track_id in wrong_ids else (0, 220, 0)
            label = f"ID {track_id} WRONG TURN" if track_id in wrong_ids else f"ID {track_id}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, label, (int(x1), max(20, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, f"Wrong-turn alerts: {len(wrong_ids)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    writer.write(annotated)
    cv2.imshow("VisionPort - Wrong Turn Detection", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
writer.release()
cv2.destroyAllWindows()
print("Wrong-turn alerts:", len(wrong_ids))
print("Output saved: outputs/wrong_turn_detection.mp4")
