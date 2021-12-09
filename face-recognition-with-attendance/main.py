import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = "attendance-images"
images =[] #for storing images
studentid= [] #id of the students could be changed to names
myls = os.listdir(path)
print(myls)

#read images of students from a database and then find their encodings
for student in myls:
    currentImage = cv2.imread(f"{path}/{student}") #providing the path
    images.append(currentImage)
    studentid.append(os.path.splitext(student)[0]) #gives us the student id

#creating a function to compute encodings
def findEncode(images):
    encodeList =[]
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(image)[0]
        encodeList.append(encodeImg)
    return encodeList

def markAttendance(name):
    with open("Attendance.csv","r+") as f:
        mylist = f.readline()
        nameList = []
        for line in mylist:
            entry = line.split(",") #split the column to name and time
            nameList.append(entry[0]) #name
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime("%H:%M:%S")#writing time
            f.writelines(f"\n{name},{dateString}")

# markAttendance("Hemsworth")

encodeStudentImages = findEncode(images)
print(len(encodeStudentImages))

#find matches in our encodings

caps = cv2.VideoCapture(0)
while True:
    success,img = caps.read()
    imgR = cv2.resize(img,(0,0),None, 0.25,0.25) #reduced size
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    #checking for locations of the face as there might be many faces
    faces = face_recognition.face_locations(imgR)
    encodeFaces = face_recognition.face_encodings(imgR,faces)

    #finding the matches, by iterating through all the faces in our frame and all the faces in our database
    for encodeFace,faceLoc in zip(encodeFaces, faces):
        #grabbing one by one
        matches = face_recognition.compare_faces(encodeStudentImages,encodeFace)
        faceDis = face_recognition.face_distance(encodeStudentImages,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if(matches[matchIndex]):
            id = studentid[matchIndex].upper()
            # print(id)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #we had scaled down the image, hence multiplying it
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,0,255),cv2.FILLED) #to display name
            cv2.putText(img, id,(x1+6,y1-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            markAttendance(id)
            break

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)