import cv2
from ultralytics import YOLO

# Load the YOLOv8 model for face detection
face_model = YOLO('yolov8n-face.pt')

# Open the video capture from the camera
cap = cv2.VideoCapture(0)

# Configura la resolución a 1080p
cap.set(3, 1920)  # Ancho
cap.set(4, 1080)  # Alto

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = face_model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Face Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
