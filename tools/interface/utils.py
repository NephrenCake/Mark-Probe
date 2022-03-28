import os
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from steganography.models.MPNet import MPEncoder, MPDecoder


def model_import(model_path, model_name, device, msg_size=96, img_size=448, warmup=True):
    """
    import encoder or decoder
    """
    if model_name == "Encoder":
        model = MPEncoder(msg_size=msg_size, img_size=img_size).to(device)
    elif model_name == "Decoder":
        model = MPDecoder(msg_size=msg_size, img_size=img_size, decoder_type="conv", has_stn=True).to(device)
    else:
        raise Exception("error! no {} model".format(model_name))

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint[model_name])
    model.eval()

    if warmup:  # 预热模型以提升后续推理速度
        if model_name == "Encoder":
            model({
                "img": torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device),
                "msg": torch.zeros((1, msg_size), dtype=torch.float32, device=device)
            })
        else:
            model(torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device))

    return model


def get_device(device):
    return torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")


def convert_img_type(img: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    将 cv2、PIL、三维Tensor 统一转换成可进入模型的四维Tensor
    todo 支持批量推理
    """
    if isinstance(img, np.ndarray):
        img = transforms.ToTensor()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img)
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        else:
            assert img.shape[0] == 1, "请勿放多张图片"
        assert img.shape[1] == 3, "传个三通道, thanks"
    return img


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor_2_cvImage(tensor_img: torch.Tensor) -> np.ndarray:
    """
    param: tensor_img 一定要是 从 pil img转过来的图片  也就是RGB图
    return: cv2 图像 BGR 格式 维度信息为 (H,W,C)
    """
    tensor_img *= 255
    if len(tensor_img.shape) == 4:
        tensor_img = tensor_img.squeeze(0)
    return cv2.cvtColor(np.uint8(tensor_img.cpu().detach().numpy()).transpose(1, 2, 0),
                        cv2.COLOR_RGB2BGR)
