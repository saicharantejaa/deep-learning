from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Restricted polygon as ratios of frame width/height (left-top, right-top,
# right-bottom, left-bottom). Change these points to fit your no-parking area.
RESTRICTED_ZONE = [(0.35, 0.45), (0.70, 0.45), (0.75, 0.90), (0.30, 0.90)]
PARKED_SECONDS = 5
MOVE_THRESHOLD_PIXELS = 5
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
zone = [(int(x * width), int(y * height)) for x, y in RESTRICTED_ZONE]
zone_contour = np.array(zone, dtype="int32")
Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/illegal_parking_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
model = YOLO("yolov8n.pt")
last_position = {}
last_movement_time = {}
alerted_ids = set()
frame_number = 0

print("Monitoring the marked restricted area. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break
    timestamp = frame_number / fps
    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=CONFIDENCE, verbose=False)[0]
    annotated = frame.copy()
    cv2.polylines(annotated, [zone_contour], True, (0, 165, 255), 2)
    cv2.putText(annotated, "RESTRICTED PARKING ZONE", (zone[0][0], max(25, zone[0][1]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            inside = cv2.pointPolygonTest(zone_contour, center, False) >= 0
            previous = last_position.get(track_id)
            if previous is not None:
                distance = ((center[0] - previous[0]) ** 2 +
                            (center[1] - previous[1]) ** 2) ** 0.5
                if distance > MOVE_THRESHOLD_PIXELS:
                    last_movement_time[track_id] = timestamp
            else:
                last_movement_time[track_id] = timestamp
            last_position[track_id] = center

            if inside:
                stopped_for = timestamp - last_movement_time.get(track_id, timestamp)
                if stopped_for >= PARKED_SECONDS:
                    alerted_ids.add(track_id)
                color = (0, 0, 255) if track_id in alerted_ids else (0, 220, 0)
                label = (f"ID {track_id} ILLEGAL PARKING" if track_id in alerted_ids
                         else f"ID {track_id} {stopped_for:.1f}s")
            else:
                color = (0, 220, 0)
                label = f"ID {track_id}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, label, (int(x1), max(20, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, f"Illegal parking alerts: {len(alerted_ids)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    writer.write(annotated)
    cv2.imshow("VisionPort - Illegal Parking", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
    frame_number += 1

video.release()
writer.release()
cv2.destroyAllWindows()
print("Illegal parking vehicle IDs:", sorted(alerted_ids))
print("Output saved: outputs/illegal_parking_detection.mp4")
