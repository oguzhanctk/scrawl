import cv2 as cv

try:
      face_cascade = cv.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
except:
      print("something went wrong while importing cascade")

image = cv.imread("resources/faces.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image, 1.1, 4)

for (x, y, w, h) in faces:
      cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))

cv.imshow("winname", image)
cv.waitKey(0)



