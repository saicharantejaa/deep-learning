from pathlib import Path
import cv2
from ultralytics import YOLO

# Demonstration settings for the junction area in the video.
MIN_VEHICLES_IN_JUNCTION = 8
ALERT_DURATION_SECONDS = 3

video_path = input("Enter the traffic-junction video path: ").strip().strip('"')
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
alert_frames_needed = int(fps * ALERT_DURATION_SECONDS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

Path("outputs").mkdir(exist_ok=True)
writer = cv2.VideoWriter("outputs/congestion_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8n.pt")
# Central/lower portion is used as a simple junction region of interest (ROI).
x1, y1 = int(width * 0.20), int(height * 0.25)
x2, y2 = int(width * 0.80), int(height * 0.95)
high_count_frames = 0
alert_active = False
frame_number = 0

print("Checking the selected junction area. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model(frame, classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
    junction_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                junction_count += 1

    if junction_count >= MIN_VEHICLES_IN_JUNCTION:
        high_count_frames += 1
    else:
        high_count_frames = 0

    congested = high_count_frames >= alert_frames_needed
    if congested and not alert_active:
        print(f"CONGESTION ALERT near video time {frame_number / fps:.1f} seconds")
    alert_active = congested

    color = (0, 0, 255) if congested else (0, 200, 0)
    label = "CONGESTION ALERT" if congested else "Traffic flowing / no alert"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"Vehicles in junction: {junction_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    writer.write(frame)
    cv2.imshow("VisionPort - Congestion Detection", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break
    frame_number += 1

video.release()
writer.release()
cv2.destroyAllWindows()
print("Processed frames:", frame_number)
print("Annotated video saved as outputs/congestion_detection.mp4")
