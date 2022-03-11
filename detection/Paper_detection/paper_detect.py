import cv2
import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(__dir__))

import utlis


def paper_find(img):
    img = cv2.resize(img, (448, 448))
    widthImg = img.shape[1]
    heightImg = img.shape[0]
    imgBigContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, dst = cv2.threshold(imgGray, thresold_value, 255, cv2.THRESH_BINARY)
    # dst = imgGray
    ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('da', th2)
    imgBlur = cv2.GaussianBlur(th2, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 125, 250)

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR

    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)

        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        return imgWarpColored, imgBigContour, biggest  # warp为透视变换后的图，bigcontour是画出的轮廓图
    else:
        point = "null"
        return img,imgBigContour,point
