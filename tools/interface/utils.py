from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from steganography.models.MPNet import MPEncoder, MPDecoder


def model_import(model_path, model_name, device, msg_size=96, img_size=448):
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

    return model


def get_device(device):
    return torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")


def convert_img_type(img: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    将 cv2、PIL、三维Tensor 统一转换成可进入模型的思维Tensor
    todo 支持批量推理
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img)
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        else:
            assert img.shape[0] == 1, "请勿放多张图片"
        assert img.shape[1] == 3, "传个三通道, thanks"
    return img
