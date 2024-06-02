from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import pickle
import os

# Capture video from the default camera (index 0)
video = cv2.VideoCapture(0)

# Load the Haar cascade for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load labels and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure FACES is correctly shaped
FACES = np.array(FACES)
print(FACES.shape)  # Check the shape to ensure correct reshaping

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgbackground=cv2.imread("bg2.jpg")

while True:
    # Read a new frame
    ret, frame = video.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame
    imgbackground[140:140 + 480,40:40 + 640] = frame
    cv2.imshow("Frame",imgbackground)

    # Check for 'q' key press to exit
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()