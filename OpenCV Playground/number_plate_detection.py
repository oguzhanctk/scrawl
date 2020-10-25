#-----> NUMBER PLATE DETECTION <-----

import cv2 as cv

#get constants variable from config file
with open("config.txt") as file:
      for line in file:
            line = line.split(" ")
            globals()[line[0]] = int(line[2])         

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, frameBrightness)
plateCascade = cv.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")

while True:
      ret, frame = cap.read()
      frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      numberPlates = plateCascade.detectMultiScale(frameGray, 1.1, 4)
      
      for (x, y, w, h) in numberPlates:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv.putText(frame, "number_plate", (x, y-5), cv.FONT_HERSHEY_COMPLEX_SMALL, \
                       1, (255, 0, 0), 2)
            cropped = frame[y:y+h, x:x+w]
            cv.imshow("plate", cropped)
            
      cv.imshow("nPlate Detection", frame)
      
      if cv.waitKey(1) & 0xFF == ord("s"):
            cv.imwrite("./scanned/plate_" + str(count) +".jpg", cropped)
            count += 1
            cv.rectangle(frame, (0, 200), (640, 300), (0, 255, 255), cv.FILLED)
            cv.putText(frame, "SCAN SAVED", (95, 265), cv.FONT_HERSHEY_DUPLEX, \
                       2, (0, 255, 0), 2)
            cv.imshow("nPlate Detection", frame)
            cv.waitKey(500)
      
      if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break


      