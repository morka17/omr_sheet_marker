import cv2
import numpy as numpy

import utils



path="001.jpg"
img = cv2.imread(path)

img_height = 700
img_width = 700

# PREPROCESSING
img  = cv2.resize(img, (img_width, img_height))
img_contours = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_canny = cv2.Canny(img_blur, 10, 50)

# FINDING COUNTOURS 
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 5)

# FIND RECTANGLES 
rectCon = utils.rectContours(contours)
biggestContour = rectCon[0]
print(len(biggestContour))

img_array = ([img, img_gray, img_blur, img_canny], 
            [img_contours, img, img, img])

image_stacked = utils.stackImages(img_array, 0.5)

cv2.imshow("Original",image_stacked)
cv2.waitKey(0)

