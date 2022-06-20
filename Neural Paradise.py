import face_recognition
import imutils
import pickle
import time
import cv2
import os
import streamlit
from faces import known_faces


cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
known_names = ['Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny',
               'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny',
               'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Mask', 'Mask', 'Mask', 'Mask', 'Mask', 'Nikita Kourov',
               'Nikita Kourov', 'Nikita Kourov', 'Nikita Kourov', 'Obabkov', 'Obabkov', 'Obabkov', 'Obabkov', 'Obabkov']
TOLERANCE = 0.6


video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    encodings = face_recognition.face_encodings(frame, model='cnn')
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_faces,
                                                 encoding, TOLERANCE)
        name = "Unknown"
        if True in matches:  # If at least one is true, get a name of first of found labels
            name = known_names[matches.index(True)]
        names.append(name)
        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
