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
        self.info = {
            "total":0,
            "right":0,
            "Hue_trans": {"total":0,"right":0,"acceptable_max":0},
            "Brightness_trans": {"total":0,"right":0,"acceptable_max":0},
            "Contrast_trans": {"total":0,"right":0,"acceptable_max":0},
            "Saturation_trans": {"total":0,"right":0,"acceptable_max":0},
            "Gaussian_blur": {"total":0,"right":0,"acceptable_max":0},  # bool
            "Rand_noise": {"total":0,"right":0,"acceptable_max":0},
            "Jpeg_trans": {"total":0,"right":0,"acceptable_max":100},  # int
            "Grayscale_trans": {"total":0,"right":0,"acceptable_max":0},  # bool
            "Rand_erase": {"total":0,"right":0,"acceptable_max":0},
            "Motion_blur": {"total":0,"right":0,"acceptable_max":0},
            "Crop": {"total":0,"right":0,"acceptable_max":0},# int
        }
        self.trans_Ops_list = [
            "Hue_trans", "Brightness_trans", "Contrast_trans",
            "Saturation_trans",  "Rand_noise",
            "Jpeg_trans", "Grayscale_trans", "Rand_erase", "Motion_blur","Crop"
        ] # "Gaussian_blur",

        self.ops_range_dict = {
            "Hue_trans": np.linspace(0, 0.3, 10),
            "Brightness_trans": np.linspace(0, 0.3, 10),
            "Contrast_trans": np.linspace(0, 0.5, 10),
            "Saturation_trans": np.linspace(0, 1, 10),
            "Gaussian_blur": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # bool
            "Rand_noise": np.linspace(0, 0.1, 10),
            "Jpeg_trans": np.linspace(10, 99, 10),  # int
            "Grayscale_trans": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # bool
            "Rand_erase": np.linspace(0, 0.1, 10),
            "Motion_blur": [i*2+1 for i in range(1,11)],
            "Crop":np.linspace(0,0.5,10)# int
        }

    def _check_img(self, img: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            return tensor_2_cvImage(img)
        if isinstance(img, np.ndarray):
            if img.dtype is float:
                raise ValueError("输入 trans 的 ndarray 数据类型为 float 达咩！！")
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
            factor = self.ops_range_dict[ops][random.randint(0, 10)]
            img_trans_order[ops] = factor
            img = self.trans(img, ops, factor)
        if _show_result_img:
            self._show_img(img, img_contrast)
        elif _show_contrast:
            self._show_img(img_contrast)

        return img, img_trans_order

    def summarize(self, img_trans_order, flag):
        self.info["total"] += 1
        if flag:
            self.info["right"] += 1
            for tag in img_trans_order:
                self.info[tag] += 1

    def summarize_single_trans(self,op_name,op_factor,flag):
        self.info[op_name]["total"]+=1
        self.info["total"]+=1
        if flag:
            self.info[op_name]["right"]+=1
            self.info["right"]+=1
            if op_name is "Jpeg_trans":
                self.info[op_name]["acceptable_max"] = min(self.info[op_name]["acceptable_max"],op_factor)
            else:
                self.info[op_name]["acceptable_max"] = max(self.info[op_name]["acceptable_max"],op_factor)
    def show_info(self):
        for tag in self.info:
            print(tag)
            print(self.info[tag])
        pass
    def clear_info(self):
        for tag in self.info:
            self.info[tag] = 0;


class SubTrans(object):
    def __init__(self):
        self.operation_dict = {
            "Hue_trans": Hue_trans(),
            "Brightness_trans": Brightness_trans(),
            "Contrast_trans": Contrast_trans(),
            "Saturation_trans": Saturation_trans(),
            "Gaussian_blur": Gaussian_blur(),
            "Rand_noise": Rand_noise(),
            "Jpeg_trans": Jpeg_trans(),
            "Grayscale_trans": Grayscale_trans(),
            "Rand_erase": Rand_erase(),
            "Motion_blur": Motion_blur(),
            "Crop":Crop(),
        }

    def __call__(self, img, ops_name, ops_factor) -> np.ndarray:
        return self.operation_dict[ops_name](img, ops_factor)
