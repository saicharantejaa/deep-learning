import cv2

image_path = input("Enter the traffic image path: ").strip().strip('"')
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or format not supported.")
    raise SystemExit

print("Image loaded successfully.")
print("Image size:", image.shape[1], "x", image.shape[0])

cv2.imshow("VisionPort - Traffic Image", image)
print("Press any key on the image window to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()
