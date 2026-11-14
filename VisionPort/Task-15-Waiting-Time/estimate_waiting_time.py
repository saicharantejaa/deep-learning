from pathlib import Path
import csv
import math
import cv2
from ultralytics import YOLO

# Maximum centre movement per frame to treat a tracked vehicle as stationary.
STATIONARY_PIXELS = 5.0

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
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
writer = cv2.VideoWriter("outputs/waiting_time_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8n.pt")
previous_centers = {}
stationary_frames = {}
maximum_wait = {}
frame_number = 0
print("Estimating how long tracked vehicles remain almost stationary. Press Q to stop.")

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

            if track_id not in stationary_frames:
                stationary_frames[track_id] = 0

            if track_id in previous_centers:
                old_center = previous_centers[track_id]
                movement = math.hypot(center[0] - old_center[0],
                                      center[1] - old_center[1])
                if movement <= STATIONARY_PIXELS:
                    stationary_frames[track_id] += 1
                else:
                    stationary_frames[track_id] = 0

            previous_centers[track_id] = center
            waiting_seconds = stationary_frames[track_id] / fps
            maximum_wait[track_id] = max(maximum_wait.get(track_id, 0), waiting_seconds)

            label = f"ID {track_id} Wait: {waiting_seconds:.1f}s"
            cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    writer.write(annotated)
    cv2.imshow("VisionPort - Waiting Time Estimate", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
    frame_number += 1

video.release()
writer.release()
cv2.destroyAllWindows()

report_path = Path("outputs/waiting_time_report.csv")
with report_path.open("w", newline="", encoding="utf-8") as report:
    writer_csv = csv.writer(report)
    writer_csv.writerow(["Vehicle ID", "Estimated maximum stationary time (seconds)"])
    for track_id, wait_seconds in sorted(maximum_wait.items()):
        writer_csv.writerow([track_id, f"{wait_seconds:.2f}"])

print("Processed frames:", frame_number)
print("Waiting-time report saved as", report_path)
print("Annotated video saved as outputs/waiting_time_video.mp4")
print("Note: These values are approximate stationary times, not guaranteed real-world queue waiting times.")
