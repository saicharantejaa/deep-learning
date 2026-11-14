import cv2

# Open the default webcam (usually camera 0).
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    raise SystemExit

print("Camera started. Point it at a traffic signal or road scene.")
print("Press S to capture and save an image. Press Q to quit.")

while True:
    success, frame = camera.read()
    if not success:
        print("Error: Could not read a frame from the camera.")
        break

    cv2.imshow("VisionPort - Camera Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") or key == ord("S"):
        saved = cv2.imwrite("traffic_image.jpg", frame)
        if saved:
            print("Image saved as traffic_image.jpg")
        else:
            print("Error: Could not save the image.")
    elif key == ord("q") or key == ord("Q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Camera closed.")
