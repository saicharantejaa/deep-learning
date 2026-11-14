from pathlib import Path
import cv2
from ultralytics import YOLO

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
vehicle_count = 0

if result.boxes is not None:
    vehicle_count = len(result.boxes)

annotated = result.plot()
cv2.putText(annotated, f"Vehicle Count: {vehicle_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

Path("outputs").mkdir(exist_ok=True)
cv2.imwrite("outputs/vehicle_count.jpg", annotated)
print("Number of vehicles in the first frame:", vehicle_count)
print("Annotated image saved as outputs/vehicle_count.jpg")

cv2.imshow("VisionPort - Vehicle Count", annotated)
print("Press any key on the image window to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()
