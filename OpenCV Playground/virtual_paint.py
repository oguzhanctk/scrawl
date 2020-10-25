import cv2 as cv
import numpy as np
from helper import findColor, drawLine

colors = [["blue", 111, 173, 137, 130, 255, 255, [255, 0, 0]], \
          ["red", 0, 152, 103, 66, 255, 255, [0, 0, 255]]]
      
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 55)
point_list = []

while True:
      ret, frame = cap.read()
      res_img = frame.copy()
      new_points = findColor(frame, colors, res_img)
      if len(new_points) != 0:
            point_list.append(new_points)
      if len(point_list) != 0:
            for p in point_list:
                  drawLine(p, res_img)
      cv.imshow("winname", res_img)
      key = cv.waitKey(1) & 0xFF
      if key == ord("q"):
            cv.destroyAllWindows()
            break
            