import cv2
import numpy as np

import os
import sys

from detection.Monitor_detection.utils.hsv_mask import hsv_range

import detection.Paper_detection.utils as utils


def find_point(img):
    width_img = img.shape[1]
    height_img = img.shape[0]

    mask = hsv_range(img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    img_blank = np.zeros((height_img, width_img, 3), np.uint8)
    max_idx = utils.max_contour_idx(contours)
    cv2.drawContours(img_blank, [contours[max_idx]], -1, (255, 255, 255), -1)
    paper = cv2.add(img, 255 - img_blank)
    mask = cv2.merge([mask, mask, mask])
    paper = cv2.add(paper, mask)

    point = utils.img_find(paper)
    if type(point) != int:
        point1 = point[0][0]
        point2 = point[1][0]
        point3 = point[2][0]
        point4 = point[3][0]
        final_img = cv2.drawContours(img, point, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        final_img = utils.drawRectangle(final_img, point, 2)
        return point1, point2, point3, point4, final_img
    else:
        img_threshold = utils.pre_treat(img)
        point = utils.canny_find(img_threshold)
        if type(point) != int:
            final_img = cv2.drawContours(img, point, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            final_img = utils.drawRectangle(final_img, point, 2)

            point1 = point[0][0]
            point2 = point[1][0]
            point3 = point[2][0]
            point4 = point[3][0]
            return point1, point2, point3, point4, final_img

