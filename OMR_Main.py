import cv2 
import numpy as np 
import utils


path = "omr2.PNG"
imgHeight = 700
imgWidth = 700


img = cv2.imread(path)
img = cv2.resize(img, (imgWidth, imgHeight))

# PRESPROCESSING 
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# FINDING ALL CONTOURS
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 4)

# FINDING RECTANGLES
rectCont = utils.rectContours(contours)
biggestContour = utils.getCornerPoints(rectCont[0])
gradePoints = utils.getCornerPoints(rectCont[4])


if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20) 


imageBlank = np.zeros_like(img)
imageArray = ([imgGray, imgBlur, imgCanny], [imgContours, imgBiggestContours, imageBlank])
imageStacked = utils.stackImages(imageArray, 0.5)

cv2.imshow("Original", imageStacked)
cv2.waitKey(0)