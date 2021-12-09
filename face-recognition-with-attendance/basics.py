import cv2
import face_recognition
import numpy as np

#loading images
face = face_recognition.load_image_file('superman.jpg')
image = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
#cvtColor is used to convert an image from one color to another
faceTest = face_recognition.load_image_file('cavill.jpg')
imageTest = cv2.cvtColor(faceTest,cv2.COLOR_BGR2RGB)

#finding the face in our image
faceLoc = face_recognition.face_locations(image)[0]
print(faceLoc) #top,right,bottom,left
#encode the face
encodeImg = face_recognition.face_encodings(image)[0]
# checking for the face and highlighting it
cv2.rectangle(image,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,0,255),2)

faceLocTest = face_recognition.face_locations(imageTest)[0]
encodeTest = face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(0,0,255),2)

#comparing faces
results = face_recognition.compare_faces([encodeImg],encodeTest)
print(results)

#checking for similarity
faceDis = face_recognition.face_distance([encodeImg],encodeTest)
#the lower the distance the better the match
print(faceDis)
cv2.putText(imageTest,f"{round(faceDis[0],2)} {results[0]}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


cv2.imshow('Image',image)
cv2.imshow('Image Test',imageTest)
cv2.waitKey(0)