from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def get_signal_colour(signal_crop):
    """Find the strongest red, yellow or green colour inside a signal crop."""
    hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)

    red_low = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_count = cv2.countNonZero(red_low) + cv2.countNonZero(red_high)

    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([90, 255, 255]))

    counts = {
        "RED": red_count,
        "YELLOW": cv2.countNonZero(yellow_mask),
        "GREEN": cv2.countNonZero(green_mask)
    }

    colour = max(counts, key=counts.get)
    if counts[colour] < 3:
        return "UNKNOWN"
    return colour


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
writer = cv2.VideoWriter("outputs/traffic_signal_status.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO("yolov8n.pt")
print("Detecting traffic lights and estimating their colour. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        break

    result = model(frame, classes=[9], conf=0.25, verbose=False)[0]  # COCO 9 = traffic light
    signal_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            status = get_signal_colour(crop)
            signal_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, status, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            print("Traffic signal status:", status)

    if signal_count == 0:
        cv2.putText(frame, "Traffic light: NOT DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    writer.write(frame)
    cv2.imshow("VisionPort - Traffic Signal Status", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
writer.release()
cv2.destroyAllWindows()
print("Annotated video saved as outputs/traffic_signal_status.mp4")
