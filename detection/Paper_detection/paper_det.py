import cv2
import numpy as np

import os
import sys

from detection.Monitor_detection.utils.hsv_mask import paper_range

import detection.Paper_detection.utils as utils


def find_point(img, num=1):
    width_img = img.shape[1]
    height_img = img.shape[0]

    mask = paper_range(img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS

    if num == 1:
        img_blank = np.zeros((height_img, width_img, 3), np.uint8)
        max_idx = utils.max_contour_idx(contours)
        cv2.drawContours(img_blank, [contours[max_idx]], -1, (255, 255, 255), -1)
        paper = cv2.add(img, 255 - img_blank)
        mask = cv2.merge([mask, mask, mask])
        paper = cv2.add(paper, mask)
        point = utils.img_find(paper)
        if type(point) != int:
            # point1 = point[0][0]
            # point2 = point[1][0]
            # point3 = point[2][0]
            # point4 = point[3][0]
            final_img = cv2.drawContours(img, point, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
            final_img = utils.drawRectangle(final_img, point, 2)
            return [[point, final_img]]
        else:
            img_threshold = utils.pre_treat(img)
            point = utils.canny_find(img_threshold)
            if type(point) != int:
                final_img = cv2.drawContours(img, point, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
                final_img = utils.drawRectangle(final_img, point, 2)

                # point1 = point[0][0]
                # point2 = point[1][0]
                # point3 = point[2][0]
                # point4 = point[3][0]
                return [[point, final_img]]
            else:
                return -1
    else:
        point_list = []
        big_contour = utils.max_contour_order(contours, 2)
        for i in range(len(big_contour)):
            area = cv2.contourArea(big_contour[i])
            # if area < 400000:
            #
            #     point_list.append([-1])
            #     continue

            mask_c = mask.copy()
            img_blank = np.zeros((height_img, width_img, 3), np.uint8)
            cv2.drawContours(img_blank, [big_contour[i]], -1, (255, 255, 255), -1)

            paper = cv2.add(img, 255 - img_blank)

            mask_merge = cv2.merge([mask_c, mask_c, mask_c])

            paper = cv2.add(paper, mask_merge)
            point = utils.img_find(paper)
            if type(point) != int:
                # point1 = point[0][0]
                # point2 = point[1][0]
                # point3 = point[2][0]
                # point4 = point[3][0]
                final_img = cv2.drawContours(img, point, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
                final_img = utils.drawRectangle(final_img, point, 2)
                point_with_img = [point, final_img]

                point_list.append(point_with_img)
            else:
                point_list.append([-1])
        return point_list

# if __name__ == "__main__":
#     img = cv2.imread('D:\Program data\pythonProject\Mark-Probe\\test\\test_img\\img_1.png')
#     img = find_point(img, 2)
#     cv2.imwrite('D:\Program data\pythonProject\Mark-Probe\out\save_img\\two_img.jpg', img)
#     cv2.waitKey(0)
