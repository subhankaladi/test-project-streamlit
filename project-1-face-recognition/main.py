import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Title
st.title("📸 Face Recognition Attendance System")

# Load known images
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Mark attendance
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{timeString}')
            st.success(f"✅ Attendance marked for {name} at {timeString}")

# Encode known faces
encodeListKnown = findEncodings(images)
st.success("✨ Encoding Complete. Ready to recognize faces!")

# Button to start webcam
if st.button("Start Webcam Attendance"):
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while True:
        success, img = cap.read()
        if not success:
            st.warning("⚠️ Could not access webcam.")
            break

        imgSmall = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgSmall)
        encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                markAttendance(name)

        stframe.image(img, channels="BGR")

        # Exit with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
