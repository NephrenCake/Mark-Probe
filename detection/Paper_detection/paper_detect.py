import cv2
import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from tools.test import hsv_range

sys.path.append(os.path.abspath(__dir__))

import utlis


def paper_find1(img):
    widthImg = img.shape[1]
    heightImg = img.shape[0]

    img2 = img.copy()
    img1 = img.copy()
    mask1 = hsv_range(img1)
    cv2.imshow('pic', mask1)
    # cv2.imshow('mask1', mask1)
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, dst = cv2.threshold(imgGray, thresold_value, 255, cv2.THRESH_BINARY)
    # dst = imgGray
    # ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('da', th2)
    # imgBlur = cv2.GaussianBlur(th2, (5, 5), 1)
    # imgThreshold = cv2.Canny(imgBlur, 125, 250)

    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    cnt = contours[max_idx]
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)

    # cv2.imshow('big', cv2.add(img1, 255 - imgBlank))
    paper = cv2.add(img1, 255 - imgBlank)
    mask1 = cv2.merge([mask1, mask1, mask1])
    paper = cv2.add(paper, mask1)
    cv2.imshow('paper',paper)
    imgwarp,imgBigContour = paper_find2(paper,img2)

    return imgwarp, imgBigContour, 'null'


def paper_find2(img,old_img):
    widthImg = img.shape[1]
    heightImg = img.shape[0]
    imgBigContour = old_img.copy()
    # imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(255 - th2, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    for factor in range(2, 20, 1):
        factor = factor / 100  # 0.002 ~ 0.2
        peri = cv2.arcLength(contours[max_idx], True) * factor
        approx = cv2.approxPolyDP(contours[max_idx], peri, True)
        if len(approx) == 4:
            biggest = utlis.reorder(approx)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 6)  # DRAW THE BIGGEST CONTOUR
            imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
            # cv2.imshow('imgblank1', imgBigContour)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(old_img, matrix, (widthImg, heightImg))
            return imgWarpColored,imgBigContour
    return img,img
# ==============================================================

# hull = cv2.convexHull(contours[max_idx])
# length = len(hull)
#
# for i in range(len(hull)):
#     cv2.line(imgBlank, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)
#
# cv2.imshow('imgblank', imgBlank)
# imgBlank = cv2.cvtColor(imgBlank, cv2.COLOR_BGR2GRAY)
# imgThreshold = cv2.Canny(imgBlank, 200, 200)
# kernel = np.ones((2, 2))
# imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
# contours, hierarchy = cv2.findContours(imgDial, cv2.RETR_EXTERNAL,
#                                        cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
# # FIND THE BIGGEST COUNTOUR
# biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
# if biggest.size != 0:
#
#     biggest = utlis.reorder(biggest)
#     cv2.drawContours(imgBlank1, biggest, -1, (0, 255, 0), 6)  # DRAW THE BIGGEST CONTOUR
#     imgBigContour = utlis.drawRectangle(imgBlank1, biggest, 2)
#     cv2.imshow('imgblank1',imgBigContour)

# ==============================================================

# FIND THE BIGGEST COUNTOUR
# biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
# peri = cv2.arcLength(contours[max_idx], True)
# approx = cv2.approxPolyDP(contours[max_idx], 0.2 * peri, True)
# print(approx)
#
# if approx.size == 8:
#     biggest = utlis.reorder(approx)
#     cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
#     imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
#     pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
#     pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
#     return imgWarpColored, imgBigContour, biggest  # warp为透视变换后的图，bigcontour是画出的轮廓图
# else:
#     point = "null"
#     return img, imgBigContour, point


# def paper_find2(img):


# def paper_find2(img):
#
#     widthImg = img.shape[1]
#     heightImg = img.shape[0]
#     imgBigContour = img.copy()
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # ret, dst = cv2.threshold(imgGray, thresold_value, 255, cv2.THRESH_BINARY)
#     dst = imgGray
#     ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # imgBlur = cv2.GaussianBlur(th2, (5, 5), 1)
#     # imgThreshold = cv2.Canny(imgBlur, 125, 250)
#
#     contours, hierarchy = cv2.findContours(255 - th2, cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
#     area = []
#     for k in range(len(contours)):
#         area.append(cv2.contourArea(contours[k]))
#     max_idx = np.argmax(np.array(area))
#     cnt = contours[max_idx]
#     imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
#     cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)
#     cv2.imshow('imgBlank',imgBlank)
#     # FIND THE BIGGEST COUNTOUR
#     # biggest, maxArea = utlis.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
#     peri = cv2.arcLength(contours[max_idx], True)
#     approx = cv2.approxPolyDP(contours[max_idx], 0.002 * peri, True)
#     print(approx.shape)
#     print(approx.size)
#     print(approx)
#     if approx.size == 0:
#         biggest = utlis.reorder(approx)
#         cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
#         imgBigContour = utlis.drawRectangle(imgBigContour, biggest, 2)
#
#         pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
#         pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
#         imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
#         return imgWarpColored, imgBigContour, biggest  # warp为透视变换后的图，bigcontour是画出的轮廓图
#     else:
#         point = "null"
#         return img, imgBigContour, point
