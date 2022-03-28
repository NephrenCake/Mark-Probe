# -- coding: utf-8 --
import math
import random

from kornia.augmentation import RandomMixUp
from kornia.filters import motion_blur
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from steganography.utils.DiffJPEG.DiffJPEG import DiffJPEG


# 注意这里是由于 之前的逻辑： 将“高分辨率图像传入并对其做失真扰乱”  而jpeg压缩对图像的大小有要求！
def jpeg_trans(img, p):
    h = img.shape[-2]
    w = img.shape[-1]
    img = F.resize(img, [(h // 16 + 1) * 16, (img.shape[-1] // 16 + 1) * 16])
    return F.resize(DiffJPEG(height=img.shape[-2], width=img.shape[-1], differentiable=True,
                             quality=random.randint(100 - int(p), 99)).to(img.device).eval()(img), [h, w])

def rand_blur(img, p):
    if p > random.uniform(0, 1):
        return img
    kernel_size = random.randint(1, 3) * 2 + 1
    t = transforms.RandomChoice([
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.GaussianBlur(kernel_size=(kernel_size, 1), sigma=(0.1, 2.0)),
        transforms.GaussianBlur(kernel_size=(1, kernel_size), sigma=(0.1, 2.0)), ])
    return t(img)


def rand_noise(img, rnd_noise):
    """
    实现随机噪声
    """
    b, c, w, h = img.size()
    noise = torch.normal(mean=0, std=rnd_noise, size=(c, w, h)).to(img.device).unsqueeze(0)
    return torch.clamp(img + noise, 0., 1.)


def rand_crop(img, scale, change_pos=False):
    """
    实现随机裁剪
        可以在画布尺寸不变的情况下，选择目标区域相对位置固定或者随机平移
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


def non_spatial_trans(img, scales):
    """
    此处主要实现大多数情况下共同包含的变换，多为非空间变换
    """
    # 反光 不使用
    if scales["reflection_trans"] != 0:
        img = RandomMixUp(p=scales["reflection_trans"], lambda_val=(0., 0.05))(img)[0]
    # 色彩
    if scales["brightness_trans"] + scales["contrast_trans"] + scales["saturation_trans"] + scales["hue_trans"] != 0:
        img = transforms.ColorJitter(brightness=scales["brightness_trans"], contrast=scales["contrast_trans"],
                                     saturation=scales["saturation_trans"], hue=scales["hue_trans"])(img)
    # 运动模糊
    # if scales["motion_blur"] > random.uniform(0, 1):
    #     img = motion_blur(img, kernel_size=2 * random.randint(1, 2) + 1,
    #                       angle=random.uniform(0, 180), direction=random.uniform(-1, 1))

    if scales["blur_trans"] != 0:
        img = rand_blur(img, scales["blur_trans"])
    # 随机噪声
    if scales["noise_trans"] != 0:
        img = rand_noise(img, scales["noise_trans"])
    # jpeg 压缩
    if int(scales["jpeg_trans"]) >= 1:
        # fit the size of JPEG_trans asked
        # img = jpeg_trans(img, scales["jpeg_trans"])
        img = DiffJPEG(height=img.shape[-2], width=img.shape[-1], differentiable=True,
                             quality=random.randint(100 - int(scales["jpeg_trans"]), 99)).to(img.device).eval()(img)
    # 灰度变换
    if scales["grayscale_trans"] != 0:
        img = transforms.RandomGrayscale(p=scales["grayscale_trans"])(img) #.to(img.device)

    return img


def rand_erase(img, _cover_rate, block_size=20):
    """
    img: torch.Tensor
    cover_rate: [0.~ 1.0)
    block_size: 遮挡块的大小 建议 0~20 pixel 规定遮挡块 是正方形
    首先将图片切分为 block_size 大小的单元格 随机填充单元格
    """
    # 在这里需要对原图进行clone操作 为啥以前不用？ md 我不理解啊！！
    # more than one element of the written-to tensor refers to a single memory location. Please clone() the tensor before performing the operation.
    _img = img.clone()
    cover_rate = random.uniform(0, _cover_rate)
    b, c, h, w = _img.shape
    block_num = [int(h * w * cover_rate) // (block_size * block_size)]
    fill_block(_img, 0, 0, w - 1, h - 1, block_num, block_size)
    while block_num[0]:
        fill_block(_img, 0, 0, w - 1, h - 1, block_num, block_size)
    return _img


def fill_block(img_, start_idx_w, start_idx_h, end_idx_w, end_idx_h, block_num, block_size):
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
    para_list = [
        [start_idx_w, start_idx_h, w_x + block_size, h_y],
        [w_x + block_size, start_idx_h, end_idx_w, h_y + block_size],
        [w_x, h_y + block_size, end_idx_w, end_idx_h],
        [start_idx_w, h_y + block_size, w_x, end_idx_h]
    ]
    idx_lis = random.sample(range(0, 4), 4)

    fill_block(img_, para_list[idx_lis[0]][0], para_list[idx_lis[0]][1], para_list[idx_lis[0]][2],
               para_list[idx_lis[0]][3], block_num, block_size)  # part 1
    fill_block(img_, para_list[idx_lis[1]][0], para_list[idx_lis[1]][1], para_list[idx_lis[1]][2],
               para_list[idx_lis[1]][3], block_num, block_size)  # part 2
    fill_block(img_, para_list[idx_lis[2]][0], para_list[idx_lis[2]][1], para_list[idx_lis[2]][2],
               para_list[idx_lis[2]][3], block_num, block_size)  # part 3
    fill_block(img_, para_list[idx_lis[3]][0], para_list[idx_lis[3]][1], para_list[idx_lis[3]][2],
               para_list[idx_lis[3]][3], block_num, block_size)  # part 4


def get_perspective_params(width: int, height: int, distortion_scale: float):
    distort_width = int(distortion_scale * (width // 2)) + 1
    distort_height = int(distortion_scale * (height // 2)) + 1

    topleft = [
        int(torch.randint(-distort_width, distort_width, size=(1,)).item()),
        int(torch.randint(-distort_height, distort_height, size=(1,)).item())
    ]
    topright = [
        int(torch.randint(width - distort_width, width + distort_width, size=(1,)).item()),
        int(torch.randint(-distort_height, distort_height, size=(1,)).item())
    ]
    botright = [
        int(torch.randint(width - distort_width, width + distort_width, size=(1,)).item()),
        int(torch.randint(height - distort_height, height + distort_height, size=(1,)).item())
    ]
    botleft = [
        int(torch.randint(-distort_width, distort_width, size=(1,)).item()),
        int(torch.randint(height - distort_height, height + distort_height, size=(1,)).item())
    ]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    return startpoints, endpoints


def make_trans_for_crop(img, scale):
    """
    该部分专门训练局部识别
    """
    # 随机裁剪
    img = rand_crop(img, scale, change_pos=False)
    return img


def make_trans_for_photo(img, scale):
    """
    该部分专门训练整体识别
    """
    # 随机块遮挡
    # if scale['erasing_trans'] != 0:
    #     img = rand_erase(img, scale['erasing_trans'], block_size=random.randint(10, 30))
    # 透视变换
    if scale['perspective_trans'] != 0:
        startpoints, endpoints = get_perspective_params(img.shape[-1], img.shape[-2], scale["perspective_trans"])
        img = F.perspective(img, startpoints, endpoints)
    return img
