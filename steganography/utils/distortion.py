# -- coding: utf-8 --
import math
import random

import torch
import torchvision.transforms.functional as transforms_F
from torchvision import transforms

from DiffJPEG.DiffJPEG import DiffJPEG


def rand_blur(img, p):
    if p < torch.rand(1):
        return img
    kernel_size = random.randint(1, 3) * 2 + 1
    t = transforms.RandomChoice([
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.GaussianBlur(kernel_size=(kernel_size, 1), sigma=(0.1, 2.0)),
        transforms.GaussianBlur(kernel_size=(1, kernel_size), sigma=(0.1, 2.0)), ])
    return t(img)


def rand_noise(img, rnd_noise):
    b, c, w, h = img.size()
    noise = torch.normal(mean=0, std=rnd_noise, size=(c, w, h)).to(img.device).unsqueeze(0)
    return torch.clamp(img + noise, 0., 1.)


def rand_crop(img, scale):
    ratio = math.cos(scale["angle_trans"] / 180 * math.pi)

    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=[1. - scale["cut_trans"], 1.],
                                                         ratio=[ratio, 1 / ratio])
    zeros = torch.zeros(img.shape)
    zeros[..., i:i + h, j:j + w] = transforms_F.crop(img, i, j, h, w)

    # img = transforms.RandomResizedCrop((w, h), scale=(1 - scale["cut_trans"], 1),
    #                                    ratio=(ratio, 1 / ratio))(img)
    return zeros


def get_custom_perspective_params(img, scale):
    b, _, w, h = img.shape

    # 确定 box
    ratio = math.cos(scale["angle_trans"] / 180 * math.pi)
    i, j, b_h, b_w = transforms.RandomResizedCrop.get_params(img, scale=[1. - scale["cut_trans"], 1.],
                                                             ratio=[ratio, 1 / ratio])

    _, startpoints = transforms.RandomPerspective.get_params(b_w, b_h, scale["perspective_trans"])
    for point in startpoints:  # bias
        point[0] += j
        point[1] += i

    boxpoints = [[j, i], [j + b_w - 1, i], [j + b_w - 1, i + b_h - 1], [j, i + b_h - 1]]
    endpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    return startpoints, boxpoints, endpoints


def make_trans(img, scale):
    b, c, w, h = img.size()

    startpoints, boxpoints, endpoints = get_custom_perspective_params(img, scale)
    no_stretched_img = img.data

    # ----------------------选择目标区域进行第一次空间变换
    use_cut = scale["perspective_trans"] + scale["angle_trans"] + scale["cut_trans"] != 0
    if use_cut:
        img = transforms_F.perspective(img, startpoints, boxpoints, fill=[0.])

    # ----------------------非空间变换
    # 色彩
    if scale["brightness_trans"] + scale["contrast_trans"] + scale["saturation_trans"] + scale["hue_trans"] != 0:
        img = transforms.ColorJitter(brightness=scale["brightness_trans"], contrast=scale["contrast_trans"],
                                     saturation=scale["saturation_trans"], hue=scale["hue_trans"])(img)
        # print("color")
    # 模糊
    if scale['blur_trans'] != 0:
        img = rand_blur(img, scale['blur_trans'])
        # print("blur_trans")
    # jpeg 压缩
    if int(scale["jpeg_trans"]) >= 1:
        img = DiffJPEG(height=h, width=w, differentiable=True,
                       quality=random.randint(100 - int(scale["jpeg_trans"]), 99)).to(img.device).eval()(img)
        # print("jpeg_trans")
    # 随机噪声
    if scale["noise_trans"] != 0:
        img = rand_noise(img, scale["noise_trans"])
        # print("noise_trans")

    # ----------------------缩放回解码器大小的空间变换
    if use_cut:
        # 实际解码图
        img = transforms_F.perspective(img, boxpoints, endpoints, fill=[0.])
        # 原图裁剪区域
        no_stretched_img = transforms_F.perspective(img, endpoints, startpoints, fill=[0.])

    return img, no_stretched_img, (torch.tensor([startpoints]) / (w - 1)).to(img.device, torch.float32).expand(b, -1,
                                                                                                               -1)

    # if scale["perspective_trans"] != 0:
    #     startpoints, endpoints = transforms.RandomPerspective.get_params(w, h, scale["perspective_trans"])
    #     img = transforms_F.perspective(img, startpoints, endpoints, fill=[0.])
    #     if cfg.perspective_no_edge:
    #         img = transforms_F.perspective(img, endpoints, startpoints, fill=[0.])
    # # 剪裁部分图片
    # if scale["angle_trans"] + scale["cut_trans"] != 0:
    #     img = rand_crop(img, scale)
    # # 随机遮挡
    # if scale["erasing_trans"] != 0:
    #     img = transforms.RandomErasing(p=1, scale=(0, scale["erasing_trans"]),
    #                                    ratio=(0.3, 3.3), value="random")(img)
