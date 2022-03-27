import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


def mask_cut(img, mask):
    origin_img = img.copy()
    # 定义 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    # 图像膨胀,确保屏幕边缘都被包括进去
    dilated_mask = cv2.dilate(mask, kernel)

    # 将图片转为灰度图
    gray_mask = cv2.cvtColor(dilated_mask, cv2.COLOR_BGR2GRAY)
    origin_img = np.array(origin_img)
    gray_mask = np.array(gray_mask)
    # 与运算
    res = cv2.bitwise_and(origin_img, origin_img, mask=gray_mask)

    return res
