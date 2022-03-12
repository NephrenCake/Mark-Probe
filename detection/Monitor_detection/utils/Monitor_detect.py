import cv2
import numpy as np
from detection.Paper_detection import utlis


def Monitor_find(old_img, img):
    # old_img = cv2.resize(old_img,(448, 448))
    # img = cv2.resize(img, (448, 448))
    # widthImg = img.shape[1]
    # heightImg = img.shape[0]

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgThreshold = cv2.Canny(imgGray, 200, 200)
    kernel = np.ones((2, 2))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    contours, hierarchy = cv2.findContours(imgDial, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(old_img, biggest, -1, (0, 255, 0), 14)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(old_img, biggest, 2)
        return imgBigContour, biggest
    else:
        point = "null"
        return old_img,point
