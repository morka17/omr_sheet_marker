import cv2
import numpy as np

import utils

#####################################
path="omr_sheet10.png"
img_height = 700
img_width = 700
choices = 5
questions = 11
#####################################


img = cv2.imread(path)


# PREPROCESSING
img  = cv2.resize(img, (img_width, img_height))
img_contours = img.copy()
img_biggestContour = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
img_canny = cv2.Canny(img_blur, 10, 50)

# FINDING COUNTOURS 
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 5)

# FIND RECTANGLES 
rectCon = utils.rectContours(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1])
print(biggestContour.shape)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(img_biggestContour, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(img_biggestContour, gradePoints, -1, (255, 0, 0,), 20)

   
    biggestContour = utils.reorder(biggestContour)
    gradePoint = utils.reorder(gradePoints)

    # Biggest contour image transformation 
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    img_warp_colored = cv2.warpPerspective(img, matrix, (img_width, img_height))

    # Grade image contour transformation
    ptG1 = np.float32(gradePoint)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])

    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    img_grade_display = cv2.warpPerspective(img, matrixG, (350, 150))
    cv2.imshow("grade", img_grade_display)

    # APPLYING A THRESHOLD
    img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
    img_threshold = cv2.threshold(img_warp_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("options", img_threshold)

    utils.splitBoxes(img_threshold)
    # pixels = np.zeros((questions, choices))
    # countCol = 0
    # countRow = 0

    # for image in boxes:
    #     totalPixels = cv2.countNonZero(image)
    #     pixels[countRow][countCol] = totalPixels
    #     countCol += 1 
    #     if countCol == choices: 
    #         countRow += 1
    #         countCol = 0 
    # print(pixels)




    

img_array = ([img, img_gray, img_blur, img_canny], 
            [img_contours, img_biggestContour, img_warp_colored, img_threshold])

image_stacked = utils.stackImages(img_array, 0.5)

cv2.imshow("Original",image_stacked)
cv2.waitKey(0)

