import numpy as np
import cv2


# img = cv2.imread('img/seg4.png')
from cv2 import CV_8U



from detection.Monitor_detection.utils.hsv_mask import monitor_hsv


def yuchuli(original_img,img,thresold_value=55):
    # img = cv2.resize(img, (512,512))
    # I = img.copy()
    # # 图像归一化
    # fI = I / 255.0
    # # 伽马变化
    # gamma = 1.5
    # img = np.power(fI, gamma)
    # img.convertTo(img, CV_8U, 255, 0)
    o_img = original_img.copy()
    dst = monitor_hsv(o_img)
    dst = 255-dst

    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlank = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # ret, dst = cv2.threshold(imgGray, thresold_value, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    res = cv2.bitwise_and(img, img, mask=dst)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(res, thresold_value, 255, cv2.THRESH_BINARY)
    # dst = cv2.erode(dst, kernel)
    k = np.ones((3, 3), np.uint8)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, k)
    # cv2.imshow('res',dst)
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
