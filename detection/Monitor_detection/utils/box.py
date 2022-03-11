import numpy as np
import cv2


# img = cv2.imread('img/seg4.png')
from cv2 import CV_8U


def yuchuli(original_img,img,thresold_value=38):
    # img = cv2.resize(img, (512,512))
    # I = img.copy()
    # # 图像归一化
    # fI = I / 255.0
    # # 伽马变化
    # gamma = 1.5
    # img = np.power(fI, gamma)
    # img.convertTo(img, CV_8U, 255, 0)
    o_img = original_img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlank = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    ret, dst = cv2.threshold(imgGray, thresold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow('s', dst)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    dst = cv2.erode(dst, kernel)

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    hull = cv2.convexHull(contours[max_idx])
    length = len(hull)
    for i in range(len(hull)):
        cv2.line(imgBlank, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)
        cv2.line(o_img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 2)

    return imgBlank,o_img


# I = img.copy()
# # 图像归一化
# fI = I / 255.0
# # 伽马变化
# gamma = 1.5
# O = np.power(fI, gamma)
# yuchuli(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
