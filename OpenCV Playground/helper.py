import cv2
import numpy as np 

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def zeroPadding(image):
      h, w, c = image.shape
      pad = np.full(((h, w + 100, 3)), 255, np.uint8)
      pad[:h, 50:pad.shape[1] - 50] = image
      cv2.resize(pad, (500, 500))
      return pad

def findColor(frame, colors, target):
      frame2hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      
      points = []
      for color in colors:
            lower = np.array(color[1:4])
            upper = np.array([color[4:7]])
            mask = cv2.inRange(frame2hsv, lower, upper)
            x, y = getContour(mask, target)
            if x != 0 and y != 0:
                  cv2.circle(target, (x, y), 15, tuple(color[7]), cv2.FILLED)
            if x != 0 and y != 0:
                  points.append((x, y, color[7]))
      return points
            
def getContour(image, res_img):
      # binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      # blur = cv2.GaussianBlur(binary, (7, 7), 1)
      # canny = cv2.Canny(blur, 150, 250)
      contours, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      x, y, w, h = 0, 0, 0, 0
      for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 170:
                  cv2.drawContours(res_img, cnt, -1, (0, 0, 0), 4)
                  peri = cv2.arcLength(cnt, False)
                  approx = cv2.approxPolyDP(cnt, 0.02*peri, False)
                  x, y, w, h = cv2.boundingRect(approx)
      # print(x+(w//2), y)
      return x+(w//2), y
      
def drawLine(points, target):
      if points:
            for point in points:
                  cv2.circle(target, (point[0], point[1]), 17, point[2], cv2.FILLED)
                  
                 
      
      
      
      
      