import cv2 
import numpy as np 
import utils


path = "omr5.png"
imgHeight = 700
imgWidth = 700
questions = 5
choices = 5
ans = [1, 1, 4, 0, 0]


img = cv2.imread(path)
img = cv2.resize(img, (imgWidth, imgHeight))

# PRESPROCESSING 
imgContours = img.copy()
final_img = img.copy()
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
gradePoints = utils.getCornerPoints(rectCont[2])
# print(biggestContour.shape)


if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20) 

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    point1 = np.float32(biggestContour)
    point2 =np.float32([[0,0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(point1, point2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))

    grade_point1 = np.float32(gradePoints)
    grade_point2 =np.float32([[0,0], [imgWidth, 0], [0, 150], [325, 150]]) 
    grade_matrix = cv2.getPerspectiveTransform(grade_point1, grade_point2)
    grade_imgWarpColored = cv2.warpPerspective(img, grade_matrix, (350, 150))
    # cv2.imshow("Grade", grade_imgWarpColored)

    # Apply Threshold
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThres = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThres)
    # cv2.imshow("Test", boxes[2])

    # GETTING NO ZERO VALUE OF EACH BOX
    _pixels_value = np.zeros((questions, choices))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        _pixels_value[countR][countC] = totalPixels
        countC += 1
        if (countC == choices):
            countR += 1
            countC = 0 
    # print(_pixels_value)

    # FINDING INDEX VALUES OF THR MARKINGS  
    myIndex = []
    for x in range(0, questions):
        arr = _pixels_value[x]
        myIndexVal = np.where(arr == np.amax(arr))
        # print(myIndex[0])
        myIndex.append(myIndexVal[0][0])
    # print(myIndex)

    # GRADING 
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else: 
            grading.append(0)
        # print(grading)

    score = sum(grading) / questions * 100  # FINAL GRADE 
    print(score)

    # DISPLAYING ANSWERS
    img_result = imgWarpColored.copy()
    img_result = utils.showAnswers(imgWarpColored, myIndex, grading , ans, questions, choices)
    img_raw_drawing = np.zeros_like(imgWarpColored)
    img_raw_drawing = utils.showAnswers(img_raw_drawing, myIndex, grading, ans, questions, choices)
    reverse_matrix = cv2.getPerspectiveTransform(point2, point1)
    rev_img_wrap = cv2.warpPerspective(img_raw_drawing, reverse_matrix, (imgWidth, imgHeight))

    img_raw_grade = np.zeros_like(grade_imgWarpColored)
    cv2.putText(img_raw_grade, str(int(score)) + "%", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255,255), 3)
    inv_matrixG = cv2.getPerspectiveTransform(point2, point1)
    inv_img_grade_display = cv2.warpPerspective(img_raw_grade, inv_matrixG, (imgWidth, imgHeight))


    final_img = cv2.addWeighted(final_img,1, rev_img_wrap, 1, 0)
    final_img = cv2.addWeighted(final_img,1, inv_img_grade_display, 1, 0)


imageBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny], 
             [imgContours, imgBiggestContours, imgWarpColored, imgThres],
             [img_result, img_raw_drawing, rev_img_wrap, final_img]
             )
imageStacked = utils.stackImages(imageArray, 0.3)


cv2.imshow("final result", final_img)
cv2.imshow("Original", imageStacked)
cv2.waitKey(0) 