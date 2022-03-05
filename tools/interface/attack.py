import copy
import random

import cv2
import numpy as np
from torchvision import transforms
from steganography.utils.DiffJPEG.DiffJPEG import DiffJPEG
import kornia

def brightness_trans(img: np.ndarray, brightness: float, gamma=0) -> np.ndarray:
    """
    传入的是cv格式的图片
    对图片进行 亮度调整 将cv图片中的每一个像素的灰度值增加1+brightness% 的量
    亮度就是每个像素所有通道都加上b
    新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    """

    im = img.astype(np.float32) * (brightness + 1)
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def contrast_trans(img: np.ndarray, contrast_factor: float) -> np.ndarray:
    """
    实现对比度的增强
    使用 线性变换 y=ax+b
    """
    contrast_factor = contrast_factor + 1
    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1 - contrast_factor) * mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def saturation_trans(img: np.ndarray, saturation_factor: float) -> np.ndarray:
    """
    饱和度变换
    """
    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1 - saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def hue_trans(img: np.ndarray, hue_factor: float) -> np.ndarray:
    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def gaussian_blur(img: np.ndarray, flag: bool) -> np.ndarray:
    """
    高斯模糊的核大小：[5,5], []
    """
    return img if flag != True else cv2.GaussianBlur(img, (5, 5), sigmaX=0.1, sigmaY=2)


def rand_noise(img: np.ndarray, std = 0.5, mean=0) -> np.ndarray:
    imgtype = img.dtype
    gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip((1 + gauss) * img.astype(np.float32), 0, 255)
    return noisy.astype(imgtype)


def jpeg_trans(img: np.ndarray, factor)->np.ndarray:
    # 将img 转换为 b x 3 x h x w
    # 添加一个判断 如果 图片的长宽高不满足要求就 用距离最近的size 进行一个resize 将resize之后 图片放入jpeg中然后 再将图片resize到原图像大小 并返回。
    w_o, h_o, _ = img.shape
    f_ = False
    if w_o % 16 or h_o % 16:
        img = cv2.resize(img, ((h_o // 16 + 1) * 16, (w_o // 16 + 1) * 16))
        f_ = True
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    b, c, h, w = img_tensor.shape
    _img = DiffJPEG(height=h, width=w, differentiable=True,
                    quality=factor)(img_tensor).squeeze(0)
    # 将一个 chw 的tensor 转换成一个 hwc的图片

    array1 = _img.detach().numpy()  # 将tensor数据转为numpy数据
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
    mat = np.uint8(array1)  # float32-->uint8
    mat = mat.transpose(1, 2, 0)  # mat_shape: (982, 814，3)
    if f_:
        mat = cv2.resize(mat, (h_o, w_o))
    # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return mat


def grayscale_trans(img: np.ndarray, flag: bool) -> np.ndarray:
    if len(img.shape) == 3 and flag:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 2 and flag:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def rand_erase(img: np.ndarray, _cover_rate:float, block_size=20) -> np.ndarray:
    im = copy.deepcopy(img)
    cover_rate = random.uniform(0, _cover_rate)
    h, w = im.shape[:2]
    block_num = [int(h * w * cover_rate) // (block_size * block_size)]
    fill_block([im], 0, 0, w - 1, h - 1, block_num, block_size)
    while block_num[0]:
        fill_block([im], 0, 0, w - 1, h - 1, block_num, block_size)
    return im


def fill_block(img_, start_idx_w, start_idx_h, end_idx_w, end_idx_h, block_num, block_size):
    # (block_num[0] <= 0)
    if (block_num[0] <= 0) or (end_idx_w - start_idx_w <= block_size) or (end_idx_h - start_idx_h <= block_size):
        return
    # 初始start_idx = 0 end_idx = w
    w_x = random.randint(start_idx_w, end_idx_w - block_size)
    h_y = random.randint(start_idx_h, end_idx_h - block_size)
    # 填充图片
    img_[0][h_y:h_y + block_size, w_x:w_x + block_size] = 0.0
    block_num[0] -= 1
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


def motion_blur(img: np.ndarray, kernel_size:int)->np.ndarray:
    '''
    方向模糊的 方向参数我直接随机了
    '''
    angle = random.uniform(0, 180)
    img = transforms.Compose([
        transforms.ToTensor()
    ])(img).unsqueeze(0)
    out = kornia.filters.motion_blur(img, kernel_size=2*kernel_size+1, angle=random.uniform(0, 180),direction=random.uniform(-1,1)).squeeze(0)
    out = (out*255.).permute(1,2,0).byte().numpy()
    return out

