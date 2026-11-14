import cv2
from pathlib import Path

# Save one frame out of every 30 frames. Change to 1 to save every frame.
FRAME_INTERVAL = 30

video_path = input("Enter the traffic video path: ").strip().strip('"')
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open the video.")
    raise SystemExit

video_name = Path(video_path).stem or "traffic_video"
output_folder = Path("extracted_frames") / video_name
output_folder.mkdir(parents=True, exist_ok=True)

frame_number = 0
saved_count = 0

while True:
    success, frame = video.read()
    if not success:
        break

    if frame_number % FRAME_INTERVAL == 0:
        file_name = output_folder / f"frame_{frame_number:06d}.jpg"
        if cv2.imwrite(str(file_name), frame):
            saved_count += 1

    frame_number += 1

video.release()
print("Total frames read:", frame_number)
print("Frames saved:", saved_count)
print("Saved in:", output_folder)
