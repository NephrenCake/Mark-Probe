import copy
import random
import cv2
from tools.interface.attack_class_rewrite import *
import numpy as np


class DistortionTestHelper():
    '''
    because the input img is a PIL.Image
    '''

    def __init__(self):
        self.trans = SubTrans()
        self.trans_Ops_list = [
            "Hue_trans","Brightness_trans","Contrast_trans",
            "Saturation_trans","Gaussian_blur","Rand_noise",
            "Jpeg_trans","Grayscale_trans","Rand_erase","Motion_blur"
        ]
        self.ops_range_dict = {
            "Hue_trans": np.linspace(0,0.1,10),
            "Brightness_trans": np.linspace(0,0.3,10),
            "Contrast_trans": np.linspace(0,0.5,10),
            "Saturation_trans": np.linspace(0,1,10),
            "Gaussian_blur": [0,0,0,0,0,1,1,1,1,1],  # bool
            "Rand_noise": np.linspace(0,0.02,10),
            "Jpeg_trans": np.linspace(50,99,10),  # int
            "Grayscale_trans": [0,0,0,0,0,1,1,1,1,1],  # bool
            "Rand_erase": np.linspace(0,0.1,10),
            "Motion_blur": np.linspace(0,10,10)  # int
        }
    def _show_img(self,img, contrast=None):
        if contrast is not None:
            cv2.imshow("t",np.hstack(img,contrast))
        else:
            cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self, img:np.ndarray, multi_times:int, _show_result_img = False, _show_contrast = False)->(np.ndarray, dict):
        idxs = random.sample(range(0,len(self.trans_Ops_list)-1),multi_times)
        img_trans_order = {}
        img_contrast = None
        if multi_times>=len(self.trans_Ops_list):
            print(f"error! the multi-trans times should be less than {multi_times}")
            return img, img_trans_order
        if _show_contrast:
            img_contrast = copy.deepcopy(img)
        for idx in idxs:
            ops = self.trans_Ops_list[idx]
            factor = self.ops_range_dict[ops][random.randint(0,9)]
            img_trans_order[ops]=factor
            img = self.trans(img,ops,factor)
        if _show_result_img:
            self._show_img(img,img_contrast)
        return img, img_trans_order



class SubTrans():
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
            "Motion_blur": Motion_blur()
        }



    def __call__(self, img, ops_name, ops_factor)->np.ndarray:
        return self.operation_dict[ops_name](img, ops_factor)


