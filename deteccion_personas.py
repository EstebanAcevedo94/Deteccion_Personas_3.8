import cv2
from ultralytics import YOLO

# Load the YOLOv8 models for person and face detection
person_model = YOLO('yolov8n.pt')
#face_model = YOLO('yolov8n-face.pt')

# Open the video capture from the camera
cap = cv2.VideoCapture(0)

# Configura la resolución a 1080p
cap.set(3, 1920)  # Ancho
cap.set(4, 1080)  # Alto

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference for person detection
        person_results = person_model(frame)

        # Run YOLOv8 for face detection
        #face_results = face_model(frame)

        # Visualize the results for person detection on the frame
        annotated_frame_person = person_results[0].plot()

        # Visualize the results for face detection on the frame
        #annotated_frame_face = face_results[0].plot()

        # Combine the two annotated frames
        #combined_annotated_frame = cv2.addWeighted(annotated_frame_person, 0.5, annotated_frame_face, 0.5, 0)

        # Display the annotated frame with both person and face detections
        cv2.imshow("Person and Face Detection", annotated_frame_person)

        # Break the loop if 'esc' is pressed
        k = cv2.waitKey(1)

        if k == 27:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
