import cv2 as cv
import numpy as np
from helper import stackImages, zeroPadding

def getContour(image):
      contours, hier = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
      for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 140:
                  cv.drawContours(res_img, cnt, -1, (0, 0, 0), 3)
                  peri = cv.arcLength(cnt, True)
                  approx = cv.approxPolyDP(cnt, 0.02*peri, True)
                  ver = len(approx)
                  x, y, w, h = cv.boundingRect(approx)
                  cv.rectangle(res_img, (x, y), (x+w, y+h), (0, 255, 0))
                  
                  if ver == 3: o_type = "triangle"
                  elif ver == 4:
                        ratio = w/float(h)
                        if 0.95 < ratio < 1.05: o_type = "square"
                        else: o_type = "rectangle"
                  elif ver > 4: o_type = "circle"
                  else: o_type = "nonetype" 
                  
                  cv.putText(res_img,
                       o_type,
                       (x+(w//2)-10,y+(h//2)-10),
                       cv.FONT_HERSHEY_COMPLEX,
                       0.7,
                       (0,0,240),
                       1)

path = "resources/shapes.jpg"


image = cv.imread(path)
image = zeroPadding(image)
res_img = image.copy()

binary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(binary, (7, 7), 1.5)
canny = cv.Canny(blur, 150, 200)

getContour(canny)
            
stc_image = stackImages(1, [image, res_img])
cv.imshow("winname", stc_image)
cv.waitKey(0)



