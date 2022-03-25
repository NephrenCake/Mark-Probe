import cv2
import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from detection.Monitor_detection.utils.hsv_mask import hsv_range

sys.path.append(os.path.abspath(__dir__))

import detection.Paper_detection.utils as utils


def paper_find(img):
    widthImg = img.shape[1]
    heightImg = img.shape[0]

    img2 = img.copy()
    img1 = img.copy()
    mask1 = hsv_range(img1)

    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    cnt = contours[max_idx]
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    cv2.drawContours(imgBlank, [cnt], -1, (255, 255, 255), -1)
    paper = cv2.add(img1, 255 - imgBlank)
    mask1 = cv2.merge([mask1, mask1, mask1])
    paper = cv2.add(paper, mask1)

    '''
    imgwarp是透视变换后的图,imgBigContour是在原图上画完轮廓后的图,point是检测到的四个点'''
    imgwarp, imgBigContour, point = img_find(paper, img2)

    if type(point) != int:
        point1 = point[0][0]
        point2 = point[1][0]
        point3 = point[2][0]
        point4 = point[3][0]
        return point1, point2, point3, point4, imgBigContour
    else:
        img_threshold = utils.pre_treat(img)
        contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        img_blank = np.zeros((heightImg, widthImg, 3), np.uint8)
        cv2.drawContours(img_blank, contours, -1, (0, 255, 0), 4)
        imgwarp, imgBigContour, four_point = utils.canny_find(img_blank)
        point1 = point[0][0]
        point2 = point[1][0]
        point3 = point[2][0]
        point4 = point[3][0]
        return point1, point2, point3, point4, imgBigContour


def img_find(img, old_img):
    widthImg = img.shape[1]
    heightImg = img.shape[0]
    imgBigContour = old_img.copy()
    # imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(255 - th2, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    max_idx = utils.max_contour_idx(contours)  # 返回最大轮廓的index

    for factor in range(2, 20, 1):
        factor = factor / 100  # 0.002 ~ 0.2
        peri = cv2.arcLength(contours[max_idx], True) * factor
        approx = cv2.approxPolyDP(contours[max_idx], peri, True)
        if len(approx) == 4:
            biggest = utils.reorder(approx)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 6)  # DRAW THE BIGGEST CONTOUR
            imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
            pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(old_img, matrix, (widthImg, heightImg))
            return imgWarpColored, imgBigContour, biggest
    return None, img, None

