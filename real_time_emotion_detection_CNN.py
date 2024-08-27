import cv2
import dlib
from joblib import load
import numpy as np

emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load the trained model
joblib_filename = './modelCNN.joblib'  # trained model file
model = load(joblib_filename)

# Set up the webcam capture and face detector
video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
FACE_SHAPE = (200, 200)  # Size of capture frame

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame for faster processing
    resized_frame = cv2.resize(frame, FACE_SHAPE)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    # Detect faces
    faces = detector(clahe_image)
    
    # Draw rectangles around detected faces and predict emotions
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = clahe_image[y:y+h, x:x+w]
        
        if face_image.size != 0:
            # Resize face image for the model
            face_image_resized = cv2.resize(face_image, (46, 46))
            image_array = np.expand_dims(face_image_resized, axis=-1) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            # Predict emotion
            predictions = model.predict(image_array)
            emotion = emotion_map[np.argmax(predictions)]
            print(emotion)

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("image", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
