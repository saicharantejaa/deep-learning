from pathlib import Path
import cv2
from ultralytics import YOLO

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
writer = cv2.VideoWriter("outputs/vehicle_tracking.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8n.pt")
print("Tracking vehicles. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                         classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
    annotated = result.plot()  # YOLO draws bounding boxes and tracking IDs.
    writer.write(annotated)
    cv2.imshow("VisionPort - Vehicle Tracking", annotated)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
writer.release()
cv2.destroyAllWindows()
print("Tracking video saved as outputs/vehicle_tracking.mp4")
