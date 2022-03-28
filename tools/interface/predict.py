from typing import Union, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from detection.Monitor_detection.deeplab import DeeplabV3
from detection.Paper_detection import paper_det
from detection.Monitor_detection.predict import predict
from steganography.models.MPNet import MPEncoder, MPDecoder
from tools.interface.bch import BCHHelper
from tools.interface.utils import convert_img_type


@torch.no_grad()
def encode(img: Union[np.ndarray, Image.Image, torch.Tensor],
           uid: Union[int, str],
           model: MPEncoder,
           bch: BCHHelper,
           device,
           img_size=(448, 448),
           direct_msg: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    你可以输入一个 cv2、PIL、三维或四维Tensor
    返回原图像大小的编码图

    现在可以同时输入一个原生 msg 以提高速度，原生 msg 可以一直保存在后端（uid+登陆时间 为定值，不需要在推理中重复构造）
        但实际提升效果不是很明显（其实不太花时间）
    """
    img = convert_img_type(img).to(device)

    if direct_msg is None:
        dat, _, _ = bch.convert_uid_to_data(uid)
        packet = torch.tensor(bch.encode_data(dat), dtype=torch.float32, device=device).unsqueeze(0)
    else:
        packet = direct_msg

    img_low = F.resize(img, img_size)
    res_low = model({"img": img_low, "msg": packet})
    res_high = F.resize(res_low, img.shape[-2:])
    encoded_img = torch.clamp(res_high + img, 0., 1.)

    return encoded_img.squeeze(0), res_low.squeeze(0)


@torch.no_grad()
def decode(img: Union[np.ndarray, Image.Image, torch.Tensor],
           bch: BCHHelper,
           model: MPDecoder,
           device,
           use_stn: bool,
           img_size=(448, 448)) -> (int, str, str, np.ndarray, float):
    img = convert_img_type(img).to(device)

    img_low = transforms.Resize(img_size)(img)
    msg_pred = model(img_low, use_stn=use_stn)[0].cpu().squeeze()
    score = torch.mean(torch.abs(0.5 - msg_pred)).item() / 0.5
    msg_pred = torch.round(msg_pred).numpy().ravel()

    bf, dat = bch.decode_data(msg_pred)
    uid, time, content = bch.convert_data_to_uid(bf, dat)

    return uid, time, content, msg_pred, score, bf


@torch.no_grad()
def detect(img: np.ndarray,
           model: DeeplabV3,
           target: str = "screen",
           thresold_1=55,
           ):
    assert target in ["screen", "paper"], "暂时只支持检测 screen 或 paper 上的隐写图像"
    if target == "screen":
        res = predict(img, model, thresold_value=thresold_1)
        if type(res) != int:
            res = [{'id': 1, 'x': res[0][0], 'y': res[0][1]},
                   {'id': 2, 'x': res[1][0], 'y': res[1][1]},
                   {'id': 3, 'x': res[2][0], 'y': res[2][1]},
                   {'id': 4, 'x': res[3][0], 'y': res[3][1]},
                   {'img': res[4]}
                   ]
            return res
        else:
            return res
    else:
        res = paper_det.find_point(img)
        if type(res) != int:
            res = [{'id': 1, 'x': res[0][0], 'y': res[0][1]},
                   {'id': 2, 'x': res[1][0], 'y': res[1][1]},
                   {'id': 3, 'x': res[2][0], 'y': res[2][1]},
                   {'id': 4, 'x': res[3][0], 'y': res[3][1]},
                   {'img': res[4]}
                   ]

            return res
        else:
            return res
