import torch
import torchvision
from PIL import Image
import random
from torchvision import transforms
import os
import cv2 as cv
import torchvision.transforms.functional as F
import numpy as np

img_save_source = 'D:\Program data\pythonProject\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\JPEGImages'
seg_save_source = 'D:\Program data\pythonProject\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\SegmentationClass'
fileDir = 'E:\dataset\\train2014'


def test_dataset():
    def get_params(img_, scale_):
        b, c, w, h = img_.shape
        print(img_.shape)
        # box
        i, j, b_h, b_w = transforms.RandomResizedCrop.get_params(img_, scale=scale_["area"], ratio=scale_["ratio"])
        # point
        _, startpoint = transforms.RandomPerspective.get_params(b_w, b_h, scale_["perspective"])
        # bias
        for point in startpoint:
            point[0] += j
            point[1] += i
        return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], startpoint

    def compose(front_name, back_name, count):
        count = str(count)
        seg_save_path = os.path.join(seg_save_source, count + '.png')
        img_save_path = os.path.join(img_save_source, count + '.jpg')
        img_front_path = os.path.join(fileDir, front_name)
        img_back_path = os.path.join(fileDir, back_name)
        img_back_size = Image.open(img_back_path).size

        trans = transforms.Compose([
            torchvision.transforms.Resize(img_back_size),
            torchvision.transforms.ToTensor()
        ])

        img_back = trans(Image.open(img_back_path).convert("RGB")).unsqueeze(0)  # 背景图
        img_front = trans(Image.open(img_front_path).convert("RGB")).unsqueeze(0)  # 前景图
        front_mask_0 = torch.ones(img_front.shape)

        i = random.randint(0, 1)  # 0 ->打印纸 1 -> 显示屏
        if i == 0:
            startpoints, endpoints = get_params(img_front, {
                "area": [0.9, 0.9],
                "ratio": [1, 1],
                "perspective": 0
            })
            img_front = F.perspective(img_front, startpoints, endpoints)
            front_mask = F.perspective(front_mask_0, startpoints, endpoints)
            img_front = torch.ones(img_front.shape) * (1 - front_mask) * (
                    np.random.randint(1, 50) / 255) + img_front  # 前景嵌入边框中

            front_mask = torch.ones(img_front.shape)

            scale = {
                "area": [0.3, 0.7],
                "ratio": [9 / 16, 16 / 9],
                "perspective": 0.2
            }

            startpoints, endpoints = get_params(img_front, scale)

            img_front = F.perspective(img_front, startpoints, endpoints)  # 对带边框的前景透视变换
            front_mask = F.perspective(front_mask, startpoints, endpoints)  # 对带边框的前景的mask透视变换
            front_mask_0 = F.perspective(front_mask_0, startpoints, endpoints)

            img = img_back * (1 - front_mask_0) + img_front  # 带边框的前景嵌入背景图
            front_mask[0, 2:, ...] *= 0

            torchvision.utils.save_image(front_mask, seg_save_path)
            torchvision.utils.save_image(img, img_save_path)
        else:
            startpoints, endpoints = get_params(img_front, {
                "area": [0.3, 0.4],
                "ratio": [1, 1],
                "perspective": 0
            })
            img_front = F.perspective(img_front, startpoints, endpoints)
            front_mask = F.perspective(front_mask_0, startpoints, endpoints)
            img_front = torch.ones(img_front.shape) * (1 - front_mask) * (np.random.randint(1, 50) / 255) + img_front

            front_mask = torch.ones(img_front.shape)

            scale = {
                "area": [0.3, 0.7],
                "ratio": [9 / 16, 16 / 9],
                "perspective": 0.2
            }

            startpoints, endpoints = get_params(img_front, scale)

            img_front = F.perspective(img_front, startpoints, endpoints)  # 对带边框的前景透视变换
            front_mask = F.perspective(front_mask, startpoints, endpoints)  # 对带边框的前景的mask透视变换
            front_mask_0 = F.perspective(front_mask_0, startpoints, endpoints)

            img = img_back * (1 - front_mask_0) + img_front  # 带边框的前景嵌入背景图
            front_mask[0, 1:, ...] *= 0

            return front_mask,img

            

    front_key = 200
    back_key = 200
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    front_img = random.sample(pathDir, front_key)  # 随机选取前景图片
    back_img = random.sample(pathDir, back_key)  # 随机获取背景图片
    for count in range(0, len(front_img)):
        front_name = front_img[count]
        back_name = back_img[count]
        compose(front_name, back_name, count)


if __name__ == "__main__":
    test_dataset()
