import cv2
import numpy as np
from helper import stackImages

#WARP PERSPECTIVE EXAMPLE
image = cv2.imread("resources/card.jpg")
width = 200
height = 300

source = np.float32([[668, 722], [1819, 22], [2035, 2173], [3203, 1304]])
dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
p_matrix = cv2.getPerspectiveTransform(source, dst)
output = cv2.warpPerspective(image, p_matrix, (width, height))

s_img = stackImages(2, [output, image])

cv2.imshow("stack image", s_img)

cv2.waitKey(0)







    




