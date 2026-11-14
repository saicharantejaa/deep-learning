from pathlib import Path
import cv2
from ultralytics import YOLO

# Simple demonstration thresholds. Adjust for the video and camera view.
LOW_MAX = 5
MEDIUM_MAX = 15

video_path = input("Enter the traffic video path: ").strip().strip('"')
if not Path(video_path).is_file():
    print("Error: Video file not found.")
    raise SystemExit

video = cv2.VideoCapture(video_path)
success, frame = video.read()
video.release()

if not success:
    print("Error: Could not read the first video frame.")
    raise SystemExit

model = YOLO("yolov8n.pt")
result = model(frame, classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
count = len(result.boxes) if result.boxes is not None else 0

if count <= LOW_MAX:
    density = "LOW"
elif count <= MEDIUM_MAX:
    density = "MEDIUM"
else:
    density = "HIGH"

annotated = result.plot()
cv2.putText(annotated, f"Vehicles: {count}  Density: {density}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/traffic_density.jpg", annotated)

print("Vehicle count:", count)
print("Traffic density:", density)
print("Annotated image saved as outputs/traffic_density.jpg")
cv2.imshow("VisionPort - Traffic Density", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
