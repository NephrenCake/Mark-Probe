from torchvision import transforms
import math
import random

import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
from steganography.config.config import TrainConfig


img = Image.open("D:\learning\COCOTrain+Val\\train2014\\COCO_train2014_000000000009.jpg")
cfg = TrainConfig()
def test1():

    trans = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Pad(padding=20, fill=125, padding_mode='constant'),
        # transforms.ToPILImage()
    ])
    img_after = trans(img).unsqueeze(0)
    img_after_ = img_after.squeeze(0)
    img_PIL = transforms.ToPILImage()(img_after_)
    print(img_PIL)
    img_PIL.show()
    print(img_after_.shape)
    print(img_after.shape)
    # img_after.show()

def test2():
    # shear 变换
    trans = transforms.Compose([
        transforms.PILToTensor(),
        transforms.RandomAffine(degrees=0,shear=cfg.shear_range),
        transforms.ToPILImage()
    ])
    img_ = trans(img)
    img_.show()

def test3():
    trans = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ColorJitter(contrast=10),
        transforms.ToPILImage()
    ])
    img_ = trans(img)
    img_.show()
    pass

def test4():
    # 翻转颜色
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter()
    ])
    pass

def test5():
    # wo tmd 要哭晕在厕所了
    trans = transforms.Compose([

        transforms.RandomEqualize(),
        transforms.ToPILImage(),


    ])

    for _ in range(4):
        img_ = F.to_tensor(img)
        img__ = trans(img_)
        img__.show()
    pass


def test6():
    trans = transforms.Compose([
        transforms.ToPILImage(),


    ])
    pass

def trans_perspective():

    pass


if __name__=="__main__":
    # test1()
    # test5()
    test1()