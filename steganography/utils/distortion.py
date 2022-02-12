# -- coding: utf-8 --
import math
import random

import torch
import torchvision.transforms.functional as F
from torchvision import transforms

from steganography.utils.DiffJPEG.DiffJPEG import DiffJPEG
from steganography.utils.MyAugmentation.transPolicy import myPolicy


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


def rand_crop(img, scale, change_pos=False):
    """
    相较于调用
        img = transforms.RandomResizedCrop((w, h), scale=(1 - scale["cut_trans"], 1), ratio=(ratio, 1 / ratio))(img)
    该方法实现，画布尺寸不变、目标区域相对位置固定，或者可以指定裁剪区域的平移后位置
    """
    ratio = math.cos(scale["angle_trans"] / 180 * math.pi)

    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=[1. - scale["cut_trans"], 1.],
                                                         ratio=[ratio, 1 / ratio])
    crop_img = torch.zeros(img.shape).to(img.device)

    if change_pos:
        new_i = torch.randint(0, img.shape[-2] - h + 1, size=(1,)).item()
        new_j = torch.randint(0, img.shape[-1] - w + 1, size=(1,)).item()
        crop_img[..., new_i:new_i + h, new_j:new_j + w] = F.crop(img, i, j, h, w)
    else:
        crop_img[..., i:i + h, j:j + w] = F.crop(img, i, j, h, w)

    return crop_img


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
    """
    todo 冲突
    """
    b, c, w, h = img.size()

    startpoints, boxpoints, endpoints = get_custom_perspective_params(img, scale)
    no_stretched_img = img.data

    # ----------------------选择目标区域进行第一次空间变换
    use_cut = scale["perspective_trans"] + scale["angle_trans"] + scale["cut_trans"] != 0
    if use_cut:
        img = F.perspective(img, startpoints, boxpoints, fill=[0.])

    # ----------------------非空间变换
    img = common_trans(img, scale)

    # ----------------------缩放回解码器大小的空间变换
    if use_cut:
        # 实际解码图
        img = F.perspective(img, boxpoints, endpoints, fill=[0.])
        # 原图裁剪区域
        no_stretched_img = F.perspective(img, endpoints, startpoints, fill=[0.])

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


def common_trans(img, scale):
    """
    此处主要实现大多数情况下共同包含的变换，多为非空间变换
    """
    b, c, w, h = img.size()

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

    return img


def rand_cover(img, _cover_rate, block_size=20):
    # img 是一个bchw的图片
    '''
    img: torch.Tensor
    cover_rate: [0.~ 1.0)
    block_size: 遮挡块的大小 建议 0~20 pixel 规定遮挡块 是正方形
    首先将图片切分为 block_size 大小的单元格 随机填充单元格
    '''
    cover_rate = random.uniform(0,_cover_rate)
    b, c, h, w = img.shape
    block_num = [int(h * w * cover_rate) // (block_size * block_size)]

    vis_w = [True] * w
    vis_h = [True] * h
    fill_block(img, 0, 0, w - 1, h - 1, block_num, block_size, vis_w, vis_h)
    # 将下面两行注释掉就实现无重叠 block覆盖 但是无法实现 指定的覆盖率
    while block_num[0]:
        fill_block(img, 0, 0, w - 1, h - 1, block_num, block_size, vis_w, vis_h)
    return img


def fill_block(img_, start_idx_w, start_idx_h, end_idx_w, end_idx_h, block_num, block_size, vis_w, vis_h):
    # (block_num[0] <= 0)
    if (block_num[0] <= 0) or (end_idx_w - start_idx_w <= block_size) or (end_idx_h - start_idx_h <= block_size):
        return
    # 初始start_idx = 0 end_idx = w
    w_x = random.randint(start_idx_w, end_idx_w - block_size)
    h_y = random.randint(start_idx_h, end_idx_h - block_size)
    # 填充图片
    img_[..., h_y:h_y + block_size, w_x:w_x + block_size] = 0.0
    block_num[0] -= 1
    # 这里涉及到 传入的img参数能否像c语言那样指定为&形式
    # torch.Tensor 是一个可变参数
    fill_block(img_, start_idx_w, start_idx_h, w_x + block_size, h_y, block_num, block_size, vis_w, vis_h)  # part 1
    fill_block(img_, w_x + block_size, start_idx_h, end_idx_w, h_y + block_size, block_num, block_size, vis_w,
               vis_h)  # part 2
    fill_block(img_, w_x, h_y + block_size, end_idx_w, end_idx_h, block_num, block_size, vis_w, vis_h)  # part 3
    fill_block(img_, start_idx_w, h_y + block_size, w_x, end_idx_h, block_num, block_size, vis_w, vis_h)  # part 4

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


def make_trans_2(img, scale):
    """
    该部分专门训练局部识别
    """
    # ----------------------非空间变换
    img = common_trans(img, scale)
    # 随机裁剪
    img = rand_crop(img, scale, change_pos=True)
    return img


def make_trans_3(img, scale):
    # 测试autoAugmentation
    if scale["myPolicy"]!=0:
        img = myPolicy()(img)
    return img

def make_trans_4(img, scale):
    """
    该部分专门训练局部识别
    """
    # ----------------------非空间变换
    img = common_trans(img, scale)
    # 随机裁剪
    img = rand_crop(img, scale, change_pos=True)
    # 随机块遮挡：
    if scale['rand_cover']!=0:
        img = rand_cover(img,scale['rand_cover'],block_size=random.randint(10, 30))
    # 透视变换
    if scale['perspective_trans']!=0:
        startpoints, endpoints = get_params(img.shape[-1], img.shape[-2], scale["perspective_trans"])
        img = F.perspective(img, startpoints, endpoints)
    return img


