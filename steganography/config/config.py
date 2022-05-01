# -- coding: utf-8 --
import math
import os
import random
import sys
import time
import numpy as np

import torch


class BaseConfig:
    def __init__(self):
        # basic config
        self.device = "cuda"
        self.seed = 2022

        # basic setting
        self.setup_seed(self.seed)

        # basic check
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.device != "cpu" else "cpu")

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


class TrainConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        self.exp_name = "lpips_dists"  # 实验名
        self.save_dir = "train_log"
        self.tensorboard_dir = "tensorboard_log"
        self.pretrained = r""
        self.resume = r""
        self.load_models = ['Encoder', 'Decoder']
        self.img_set_list = {
            r"D:\服务外包大赛_project\COCo\train2014\train2014\train2014": 1,
            r"D:\服务外包大赛_project\COCo\val2014\val2014\val2014\val2014": 1,
        }
        self.val_rate: float = 0.05  # 用于验证的比例
        self.log_interval = 200  # 打印日志间隔 iterations
        self.img_interval = 1000  # 保存图像间隔 iterations

        self.max_epoch = 10  # 训练的总轮数
        self.batch_size = 4  # 一个批次的图片数量
        self.num_workers = 8  # 进程数
        self.single = True  # 是否多卡训练  False：使用多卡

        self.weight_decay = 0
        self.lr_base = 1e-5  # 基础学习率
        self.lr_max = 1  # 最高学习率倍率
        self.lr_min = 0.1  # 最低学习率倍率
        self.warm_up_epoch = 0  # 完成预热的轮次
        self.use_warmup = True

        # ============== module
        self.img_size = (448, 448)  # 输入网络的图片大小  注意，只能正方形
        self.msg_size = 96  # 输入网络的二进制字符串大小

        # ============== dynamic scales
        # 注册使用的递增变换
        self.scale_list = [
            "grayscale_trans", "motion_blur", "clamp_limit",
            "perspective_trans", "angle_trans", "cut_trans", "erasing_trans", "jpeg_trans", "noise_trans",
            "brightness_trans", "contrast_trans", "saturation_trans", "hue_trans", "blur_trans", "reflection_trans",
            "rgb_loss", "hsv_loss", "yuv_loss", "lpips_loss", 'stn_loss', "dists_loss"
        ]

        # res clamp limit
        self.clamp_limit_max = 0
        self.clamp_limit_grow = (2, 5)

        # transform scale
        # (epochA, epochB) 代表 epochA -> epochB 的权重递增
        self.perspective_trans_max = 0.05  # 透视变换
        self.perspective_trans_grow = (2, 5)
        self.angle_trans_max = 30  # 观察图片的视角，指与法线的夹角，入射角
        self.angle_trans_grow = (1, 1.8)
        self.cut_trans_max = 0.5  # 舍弃的图片区域
        self.cut_trans_grow = (1, 1.8)

        self.jpeg_trans_max = 50  # 这里表示压缩强度。而图像质量是   上调 <= 70
        self.jpeg_trans_grow = (0, 0.1)
        # self.motion_blur_max = 0  # 给出的motion——blur的 核的最大值 （按照 2*k+1方式）  这个最大值是 实际中运动模糊实际的随机取值的最大值。
        # self.motion_blur_grow = (1.5, 2)
        self.blur_trans_max = 0.2
        self.blur_trans_grow = (0, 0.1)

        self.reflection_trans_max = 0.00  # 反光的概率
        self.reflection_trans_grow = (0.6, 0.9)
        self.grayscale_trans_max = 0.1
        self.grayscale_trans_grow = (0.6, 0.9)
        self.erasing_trans_max = 0  # 随机遮挡
        self.erasing_trans_grow = (0.6, 0.9)
        self.noise_trans_max = 0.02
        self.noise_trans_grow = (0.6, 0.9)

        self.brightness_trans_max = 0.3  # 亮度变换
        self.brightness_trans_grow = (0.3, 0.9)
        self.contrast_trans_max = 0.5  # 对比度变换
        self.contrast_trans_grow = (0.3, 0.9)
        self.saturation_trans_max = 1  # 饱和度变换
        self.saturation_trans_grow = (0.3, 0.9)
        self.hue_trans_max = 0.1  # 色相变换
        self.hue_trans_grow = (0.3, 0.9)

        # loss scale
        self.rgb_loss_max = 1
        self.rgb_loss_grow = None
        self.hsv_loss_max = 0
        self.hsv_loss_grow = None
        self.yuv_loss_max = 1
        self.yuv_loss_grow = None
        self.lpips_loss_max = 1
        self.lpips_loss_grow = (0, 0.5)
        self.dists_loss_max = 1
        self.dists_loss_grow = (0, 0.5)
        # other
        self.stn_loss_max = 1  # 换成1时可以开启，0则不对stn进行训练
        self.stn_loss_grow = (1.8, 1.8)

        # ============== runtime
        self.iter_per_epoch = None
        self.start_save_best = 3  # 直到grow停止时才保存最优模型
        self.scale_name_list = [name.split("_grow")[0] for name in vars(self).keys() if "_grow" in name]

    def check_cfg(self):
        self.exp_name = self.exp_name + time.strftime('_%Y-%m-%d-%H-%M-%S', time.localtime())
        self.num_workers = min([self.batch_size if self.batch_size > 1 else 0, self.num_workers, os.cpu_count(), 16]) \
            if "win" not in sys.platform else 0
        if len(self.resume) != 0:
            self.exp_name = self.resume.split("/")[-2]
        for name, value in vars(self).items():
            if name.endswith("grow") and value is not None and value[1] > self.start_save_best:
                self.start_save_best = value[1]
        _check_dir(os.path.join(sys.path[0], self.save_dir, self.exp_name))

    def get_cur_scale(self, scale_name: str, cur_iter: int, cur_epoch: int):
        scale_max = eval("self." + scale_name + "_max")
        scale_grow = eval("self." + scale_name + "_grow")
        global_pos = (cur_iter + 1) / self.iter_per_epoch + cur_epoch
        if scale_grow is None:
            return scale_max
        elif scale_grow[1] == scale_grow[0]:
            return scale_max if global_pos > scale_grow[0] else 0
        else:
            return max(min(scale_max * (global_pos - scale_grow[0]) / (scale_grow[1] - scale_grow[0]), scale_max), 0)

    def get_cur_scales(self, cur_iter: int, cur_epoch: int, scale_name_list=None):
        if scale_name_list is None:
            scale_name_list = self.scale_name_list
        cur_scales = {}
        for scale_name in scale_name_list:
            if scale_name not in self.scale_list:
                cur_scales[scale_name] = 0
            else:
                cur_scales[scale_name] = self.get_cur_scale(scale_name, cur_iter, cur_epoch)
        return cur_scales

    def set_iteration(self, iteration: int):
        self.iter_per_epoch = iteration
        # self.iter_total = iteration * self.max_epoch

    def get_warmup_cos_lambda(self, start_iter=0):
        warm_up_iter = self.warm_up_epoch * self.iter_per_epoch
        t_max = (self.max_epoch - self.warm_up_epoch - start_iter) * self.iter_per_epoch

        def warmup_cos_lambda(cur_iter):
            if cur_iter < start_iter:
                return 0
            elif cur_iter < warm_up_iter + start_iter:
                return (cur_iter - start_iter) / warm_up_iter * (self.lr_max - self.lr_min) + self.lr_min
            elif cur_iter < self.max_epoch * self.iter_per_epoch:
                return (self.lr_max - self.lr_min) * (1.0 + math.cos((cur_iter - start_iter - warm_up_iter) /
                                                                     t_max * math.pi)) / 2 + self.lr_min
            else:
                return self.lr_min

        return warmup_cos_lambda

    def get_global_iter(self, cur_epoch, cur_iter):
        return cur_epoch * self.iter_per_epoch + cur_iter


def _check_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
