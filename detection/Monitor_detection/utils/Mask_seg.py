import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


# png = cv2.imread('./img_single/test2.jpeg')
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(png)
# plt.axis('off')
#
# # 定义 kernel
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
#
# # # 图像腐蚀
# # eroded = cv2.erode(png, kernel)
# # plt.subplot(1, 3, 2)
# # plt.imshow(eroded) # Eroded Image
# # plt.axis('off')
#
# # 图像膨胀
# dilated = cv2.dilate(png, kernel)
# plt.subplot(1, 3, 2)
# plt.imshow(dilated)  # Dilated Image
# plt.axis('off')
#
# old_img = cv2.imread('./img/seg2.png')
# src_img = dilated
# origin_img = old_img
# old_img = Image.fromarray(np.uint8(old_img))
# dilated = Image.fromarray(np.uint8(dilated))
# image = Image.blend(old_img, dilated, 0.7)
# plt.subplot(1, 3, 3)
# plt.imshow(image)  # Dilated Image
# plt.axis('off')
# plt.show()
#
# # 将图片转为灰度图
# gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#
# # retval, dst = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
# # 最大类间方差法(大津算法)，thresh会被忽略，自动计算一个阈值
# retval, dst = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#
# # 与运算
# res = cv2.bitwise_and(origin_img, origin_img, mask=dst)
# cv2.imwrite('img/seg4.png', res)
# cv2.imshow('s', res)
# cv2.waitKey(0)


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


# if __name__ == '__main__':
#     img = cv2.imread('./img/img_6.png')
#     mask = cv2.imread('img_single/img_6.png')
#     mask_cut(img, mask)
