import cv2
import numpy as np



def stackImages(imageArray, scale, lables=[]):
    rows = len(imageArray)
    cols = len(imageArray[0])
    rowsAvailable = isinstance(imageArray[0], list)
    width = imageArray[0][0].shape[0]
    height = imageArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imageArray[x][y] = cv2.resize(imageArray[x][y], (0, 0), None, scale, scale)
                if len(imageArray[x][y].shape) == 2: 
                    imageArray[x][y] = cv2.cvtColor(imageArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]  *rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imageArray[x])
            hor_con[x] = np.concatenate(imageArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imageArray[x] = cv2.resize(imageArray[x], (0,0), None, scale, scale)
            if len(imageArray[x].shape) == 2: 
                imageArray[x] = cv2.cvtColor(imageArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imageArray)
        hor_con = np.concatenate(imageArray)
        ver = hor 
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c* eachImgWidth, eachImgWidth * d), (c* eachImgWidth + len(lables[d][c]) * 13+27, 30+eachImgHeight))
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20 ), cv2.FONT_HERSHEY_COMPLEX, 0.7)

    return ver     




def rectContours(contours):

    rectCont = []
    
    for i in contours:
        area = cv2.contourArea(i)
        # print(area)
        if area > 50: 
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            # print("Area ", area, "Corner Points", len(approx))
            if len(approx) == 4: 
                rectCont.append(i)
    rectCont = sorted(rectCont, key= cv2.contourArea, reverse = True)

    return rectCont



def getCornerPoints(cont):
    perimeter = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * perimeter, True)
    return approx



def reorder(points):
    _points = points.reshape((4, 2))
    newPoint = np.zeros((4, 1, 2), np.int32)
    add = _points.sum(1)
    newPoint[0] = _points[np.argmin(add)] #[0, 0]
    newPoint[3] = _points[np.argmax(add)] # [w, h]
    diff = np.diff(_points, axis=1)
    newPoint[1] = _points[np.argmin(diff)] # [w, 0]
    newPoint[2] = _points[np.argmax(diff)] # [0, h]
    
    return newPoint


def splitBoxes(img):
    rows = np.vsplit(img, 5)
    cv2.imshow("Split", rows[0])
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 5)
        for box in cols: 
            boxes.append(box)
            # cv2.imshow("option1", box)
    # return boxes
     



# FIND THE CENTER VALUE OF A GIVEN BLOCK
def showAnswers(img, index, grading , ans, questions, choices):
    secW =int(img.shape[1] / questions )
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        answer = index[x]
        cX = (answer  *  secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x] == 1:
            _color = (0, 255, 0)
        else:
            _color = (0, 0, 255)
            correct_answer = ans[x]
            cv2.circle(img, ((correct_answer * secW) + secW // 2, (x * secH) + secH // 2), 20, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 50, _color, cv2.FILLED)
    return img  