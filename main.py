import cv2
from random import randrange
#load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#choose an image to detect the face
img = cv2.imread("people.jpg")

#changing to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

#draw rectangle
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y),(x+w, y+h), (randrange(256),randrange(256),randrange(256)),3) #here 255 is green and 2 is the thickness of the rectangle

#print(face_codinates)

# to show the image
cv2.imshow("Ansub Khan Face Detector",img)
cv2.waitKey()  #pauses the execution of the above line

print("This is Ansub Khan")
