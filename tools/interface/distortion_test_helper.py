import copy
import os
import random
from typing import Union

import cv2
import torch

from tools.interface.attack_class_rewrite import *
import numpy as np
from tools.interface.utils import tensor_2_cvImage


class DistortionTestHelper(object):
    '''
    because the input img is a PIL.Image
    '''

    def __init__(self):
        self.trans = SubTrans()
        self.trans_Ops_list = [
            "Hue_trans", "Brightness_trans", "Contrast_trans",
            "Saturation_trans",  "Rand_noise",
            "Jpeg_trans", "Grayscale_trans", "Rand_erase", "Motion_blur"
        ] # "Gaussian_blur",

        self.ops_range_dict = {
            "Hue_trans": np.linspace(0, 0.1, 10),
            "Brightness_trans": np.linspace(0, 0.3, 10),
            "Contrast_trans": np.linspace(0, 0.5, 10),
            "Saturation_trans": np.linspace(0, 1, 10),
            # "Gaussian_blur": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # bool
            "Rand_noise": np.linspace(0, 0.02, 10),
            "Jpeg_trans": np.linspace(50, 99, 10),  # int
            "Grayscale_trans": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # bool
            "Rand_erase": np.linspace(0, 0.1, 10),
            "Motion_blur": np.linspace(0, 10, 10)  # int
        }

    def _check_img(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            return tensor_2_cvImage(img)
        if isinstance(img, np.ndarray):
            if img.dtype is float:
                raise ValueError("输入 trans 的 ndarray数据类型为 float 达咩！！")
            return img
        else:
            raise ValueError("输入的img 类型不是 torch.Tensor or np.ndarray!! 请检查")

    def _show_img(self, img: np.ndarray, contrast=None):
        if contrast is not None:
            if img.shape != contrast.shape:
                raise ValueError("contrast 图片和 img 图片 size 不一致")
            cv2.imshow("left: transformed , right: origin  ", np.hstack((img, contrast)))
        else:
            cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self, img: Union[torch.Tensor, np.ndarray], multi_times: int, _show_result_img=False,
                 _show_contrast=False) -> (
            np.ndarray, dict):
        idxs = random.sample(range(0, len(self.trans_Ops_list) - 1), multi_times)
        img_trans_order = {}
        img_contrast = None
        if multi_times >= len(self.trans_Ops_list):
            return img, img_trans_order
        img = self._check_img(img)
        if _show_contrast:
            img_contrast = copy.deepcopy(img)
        for idx in idxs:
            ops = self.trans_Ops_list[idx]
            factor = self.ops_range_dict[ops][random.randint(0, 9)]
            img_trans_order[ops] = factor
            img = self.trans(img, ops, factor)
        if _show_result_img:
            self._show_img(img, img_contrast)

        return img, img_trans_order


class SubTrans(object):
    def __init__(self):
        self.operation_dict = {
            "Hue_trans": Hue_trans(),
            "Brightness_trans": Brightness_trans(),
            "Contrast_trans": Contrast_trans(),
            "Saturation_trans": Saturation_trans(),
            # "Gaussian_blur": Gaussian_blur(),
            "Rand_noise": Rand_noise(),
            "Jpeg_trans": Jpeg_trans(),
            "Grayscale_trans": Grayscale_trans(),
            "Rand_erase": Rand_erase(),
            "Motion_blur": Motion_blur()
        }

    def __call__(self, img, ops_name, ops_factor) -> np.ndarray:
        return self.operation_dict[ops_name](img, ops_factor)
