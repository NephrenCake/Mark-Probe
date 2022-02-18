import os
import sys

import cv2

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from steganography.utils.distortion import rand_crop, grayscale_trans

img_path = "test_source/COCO_train2014_000000000009.jpg"
img_path = "D:\learning\COCOTrain+Val\\test2014\COCO_test2014_000000000001.jpg"
img_size = (448, 448)
msg_size = 96
scale = {
    "angle_trans": 30,
    "cut_trans": 0.5,
    "perspective_trans": 0.1,
}

img = transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor()
])(Image.open(img_path).convert("RGB")).unsqueeze(0)
torchvision.utils.save_image(img, "test_source/test.jpg")

def show_img(_img: torch.Tensor):
    '''
    展示一张tensor形式的图片
    '''
    img_ = transforms.ToPILImage()(_img)
    img_.show()
    pass

def test_crop():
    global img
    img_ = img.clone().detach()

    img_ = rand_crop(img_, scale, change_pos=False)
    torchvision.utils.save_image(img_, "test_source/test_crop.jpg")


def test_perspective():
    def get_params(width: int, height: int, distortion_scale: float):
        distort_width = int(distortion_scale * (width // 2)) + 1
        distort_height = int(distortion_scale * (height // 2)) + 1

        topleft = [
            int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
            int(torch.randint(-distort_height, distort_height, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
            int(torch.randint(-distort_height, distort_height, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    global img
    img_ = img.clone().detach()

    startpoints, endpoints = get_params(img_size[0], img_size[0], scale["perspective_trans"])
    img_ = F.perspective(img_, startpoints, endpoints)
    torchvision.utils.save_image(img_, "test_source/test_perspective.jpg")

def get_params(width: int, height: int, distortion_scale: float):
    distort_width = int(distortion_scale * (width // 2)) + 1
    distort_height = int(distortion_scale * (height // 2)) + 1

    topleft = [
        int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
        int(torch.randint(-distort_height, distort_height, size=(1, )).item())
    ]
    topright = [
        int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
        int(torch.randint(-distort_height, distort_height, size=(1, )).item())
    ]
    botright = [
        int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
        int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
    ]
    botleft = [
        int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
        int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
    ]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    return startpoints, endpoints

def test1():
    startpoints, endpoints = get_params(img_size[0], img_size[0], scale["perspective_trans"])
    img_ = F.perspective(img, startpoints, endpoints).squeeze(0)
    show_img(img_)
    pass

def test_grayscale():
    print(img.shape[:])
    # img_ = transforms.Grayscale()(img).squeeze(0)
    # 使用luma原理来实现grayscale变换
    # print(img_.shape)
    # show_img(img_)
    # 输入一个 b 3 h w 的图片组 通过grayscale 会变为 b 1 h w的图片组 c通道少了俩 grayscale 保留的是哪一个通道的灰度图?
    # luma L = R * 299/1000 + G * 587/1000 + B * 114/1000，下取整
    # b,c,h,w = img.shape
    # for i in range(0,b):
    #     img[i][0] = img[i][0]*0.299+img[i][1]*0.587+img[i][2]*0.114
    #     img[i][1] = img[i][0]
    #     img[i][2] = img[i][0]
    # img_ = img.squeeze(0)
    # print(img_.shape)
    # print(img.shape)
    # show_img(img.squeeze(0))
    # 测试img 是否直接被 函数修改 应该是直接被修改了 tensor是个可变参数
    grayscale_trans(img,0)
    show_img(img.squeeze(0))

    pass

def test_contrast():
    _img = F.adjust_contrast(img,contrast_factor=0.4).squeeze(0)
    show_img(_img)


def test_gray_trans():
    _img = cv2.imread(img_path)
    _img_gray = cv2.cvtColor(_img,cv2.COLOR_BGR2GRAY)
    print(_img.shape)
    print(_img[0:10][0:10][1])
    print(_img_gray[0:10][0:10])



if __name__ == '__main__':
    # test_crop()
    # test_perspective()
    # test1()
    test_grayscale()
    # test_contrast()
    # test_gray_trans()
