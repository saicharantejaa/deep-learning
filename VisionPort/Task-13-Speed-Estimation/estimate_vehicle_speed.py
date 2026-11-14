from pathlib import Path
import math
import cv2
from ultralytics import YOLO

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

try:
    meters_per_pixel = float(input(
        "Enter camera calibration (metres per pixel, e.g. 0.05): "))
except ValueError:
    print("Error: Enter a numeric calibration value.")
    raise SystemExit

if meters_per_pixel <= 0:
    print("Error: Calibration must be greater than zero.")
    raise SystemExit

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error: Could not open the video.")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/speed_estimation.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8n.pt")
previous_positions = {}
frame_number = 0
print("Speed is estimated from tracked-centre movement. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
    annotated = result.plot()

    if result.boxes is not None and result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
        coordinates = result.boxes.xyxy.cpu().tolist()

        for track_id, box in zip(ids, coordinates):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            label = f"ID {track_id}: measuring"

            if track_id in previous_positions:
                old_center, old_frame = previous_positions[track_id]
                frame_gap = frame_number - old_frame
                if frame_gap > 0:
                    distance_pixels = math.hypot(
                        center[0] - old_center[0], center[1] - old_center[1]
                    )
                    speed_mps = (distance_pixels / frame_gap) * fps * meters_per_pixel
                    speed_kmh = speed_mps * 3.6
                    label = f"ID {track_id}: {speed_kmh:.1f} km/h"

            previous_positions[track_id] = (center, frame_number)
            cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    writer.write(annotated)
    cv2.imshow("VisionPort - Approximate Speed", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
    frame_number += 1

video.release()
writer.release()
cv2.destroyAllWindows()
print("Processed frames:", frame_number)
print("Annotated video saved as outputs/speed_estimation.mp4")
