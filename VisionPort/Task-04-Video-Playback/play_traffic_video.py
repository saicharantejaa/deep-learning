import cv2

video_path = input("Enter the traffic video path: ").strip().strip('"')
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Could not open the video.")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25
wait_time = max(1, int(1000 / fps))

print("Video playing. Press Q to stop.")

while True:
    success, frame = video.read()
    if not success:
        print("Video finished.")
        break

    cv2.imshow("VisionPort - Traffic Video", frame)
    if cv2.waitKey(wait_time) & 0xFF in (ord("q"), ord("Q")):
        break

video.release()
cv2.destroyAllWindows()
