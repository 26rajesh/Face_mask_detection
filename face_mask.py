import cv2
from keras.models import load_model
import numpy as np
import imutils

face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

model = load_model('model.h5')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    frame = cam.read()[1]
    height, weight = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    labels = ['With Mask','Without Mask']

    faces = face.detectMultiScale(gray, minNeighbors = 5, scaleFactor = 1.1, minSize = (25,25))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (24,24))
        face_roi = face_roi/255
        face_roi = np.expand_dims(face_roi, axis = -1)
        face_roi = np.expand_dims(face_roi, axis = 0)
        pred = model.predict(face_roi)[0]
        label_idx = np.argmax(pred)
        label = labels[label_idx]
    cv2.putText(frame, label, (x,y-10), font, 1, (0,255,0),2)

    cv2.imshow('Face Mask Detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
