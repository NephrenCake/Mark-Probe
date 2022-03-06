# -- coding: utf-8 --
import logging
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from kornia.enhance import normalize_min_max, AdjustGamma
from kornia.filters import laplacian
from torch.utils.data import Dataset
from torchvision import transforms


def get_dataloader(img_set_list: dict,
                   img_size: tuple,
                   msg_size: int,
                   val_rate: float,
                   batch_size: int,
                   num_workers: int):
    # todo 加入保存数据集缓存
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]  # 支持的文件后缀类型

    dataset_len = []  # 使用数据集总量
    train_img_list = []  # 训练集的所有图片路径
    val_img_list = []  # 验证集的所有图片路径

    for img_dir in img_set_list:
        images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if os.path.splitext(i)[-1] in supported]
        use_rate = img_set_list[img_dir]

        use_num = int(len(images) * use_rate)
        val_num = int(use_num * val_rate)
        images_used = random.sample(images, k=use_num)  # 对每个数据集采样使用数量
        images_val = random.sample(images_used, k=val_num)  # 对确定适用的数据集采样验证集数量

        dataset_len.append(use_num)
        for img_path in images_used:
            if img_path in images_val:
                val_img_list.append(img_path)
            else:
                train_img_list.append(img_path)

    logging.info("{} images were in used.".format(sum(dataset_len)))
    logging.info("{} images for training.".format(len(train_img_list)))
    logging.info("{} images for validation.".format(len(val_img_list)))

    # 定义训练以及预测时的预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size, scale=(0.5, 1)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                     transforms.RandomGrayscale(p=0.05),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   ]),
    }

    # 实例化数据集
    train_data_set = StegaDataset(img_list=train_img_list,
                                  msg_size=msg_size,
                                  transform=data_transform["train"])
    val_data_set = StegaDataset(img_list=val_img_list,
                                msg_size=msg_size,
                                transform=data_transform["val"])
    # 实例化加载器
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    return train_loader, val_loader


class StegaDataset(Dataset):
    def __init__(self, img_list: list, msg_size: int, transform=None):
        self.img_list = img_list
        self.msg_size = msg_size
        self.transform = transform
        self.adjustGamma = AdjustGamma(10., 1.)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_list[idx]).convert("RGB")
        except:
            logging.warning(f"read {self.img_list[idx]} failed.")
            img = generate_random_circle_picture()
        if self.transform is not None:
            img = self.transform(img)

        msg = np.random.binomial(1, .5, self.msg_size)
        msg = torch.from_numpy(msg).to(torch.float)

        # mask for loss function
        # 现在 mask 是一个范围 [0., 1.] 边缘区域像素值低 平滑区域像素值高
        mask = 1 - torch.abs(laplacian(img.unsqueeze(0), 3))  # low weight in high frequency
        mask = self.adjustGamma(normalize_min_max(mask)).squeeze(0)

        return {
            "img": img,
            "msg": msg,
            "mask": mask,
        }


def generate_random_circle_picture(d=500):
    img = np.zeros((d, d, 3), dtype=np.float32)
    for i in range(0, 50):
        center_x = np.random.randint(0, d)
        center_y = np.random.randint(0, d)
        radius = np.random.randint(5, d / 5)
        color = np.random.randint(0, 256, size=(3,)).tolist()
        cv2.circle(img, (center_x, center_y), radius, color, -1)
    return Image.fromarray(img, mode='RGB')
