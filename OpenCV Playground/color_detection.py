import cv2
import numpy as np
from helper import stackImages

def empty(a):
      pass
#hue, saturation, val

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 480)

cv2.createTrackbar("min_hue", "Trackbars", 0, 179, empty)
cv2.createTrackbar("max_hue", "Trackbars", 34, 179, empty)
cv2.createTrackbar("min_sat", "Trackbars", 0, 255, empty)
cv2.createTrackbar("max_sat", "Trackbars", 100, 255, empty)
cv2.createTrackbar("min_val", "Trackbars", 0, 255, empty)
cv2.createTrackbar("max_val", "Trackbars", 110, 255, empty)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:      
      # image = cv2.imread("resources/shapes.jpg")
      # image = cv2.resize(image, (620, 480))
      ret, image = cap.read()
      img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      h_min = cv2.getTrackbarPos("min_hue", "Trackbars") 
      h_max = cv2.getTrackbarPos("max_hue", "Trackbars") 
      s_min = cv2.getTrackbarPos("min_sat", "Trackbars") 
      s_max = cv2.getTrackbarPos("max_sat", "Trackbars") 
      v_min = cv2.getTrackbarPos("min_val", "Trackbars") 
      v_max = cv2.getTrackbarPos("max_val", "Trackbars") 
      
      lower = np.array([h_min, s_min, v_min])
      upper = np.array([h_max, s_max, v_max])
      print(lower)
      print(upper)
      mask = cv2.inRange(img_hsv, lower, upper)
      img_res = cv2.bitwise_and(image, image, mask=mask)

      stacked_images = stackImages(0.5, ([mask, img_res], [image, img_hsv]))
      
      
      cv2.imshow("res", stacked_images)
      cv2.waitKey(1)