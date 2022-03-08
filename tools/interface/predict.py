from typing import Union, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from detection.Monitor_detection.deeplab import DeeplabV3
from detection.Paper_detection import paper_detect
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
           img_size=(448, 448)) -> (torch.Tensor, torch.Tensor):
    img = convert_img_type(img).to(device)

    img_low = transforms.Resize(img_size)(img)
    dat, now, key = bch.convert_uid_to_data(uid)
    packet = torch.tensor(bch.encode_data(dat), dtype=torch.float32, device=device).unsqueeze(0)  # 数据段+校验段

    res_low = model({"img": img_low, "msg": packet})
    res_high = F.resize(res_low, img.shape[-2:])
    encoded_img = torch.clamp(res_high + img, 0., 1.)

    return encoded_img.squeeze(0).cpu(), res_low.squeeze(0).cpu()


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
def detect(img: Union[np.ndarray, Image.Image, torch.Tensor],
           target: str = "screen") -> List[List]:
    deeplab = DeeplabV3()
    assert target in ["screen", "paper"], "暂时只支持检测 screen 或 paper 上的隐写图像"
    if target == "screen":
        final_img, point = predict(img,deeplab)
        return [final_img, point]
        # 返回标注好点的图片以及四个点的坐标,取四个点的坐标时可写为point[0][0],point[0][1],point[0][2],point[0][3]
    else:
        paper_img, point1 = paper_detect.paper_find(img)
        img_on_paper, point2 = paper_detect.paper_find(paper_img)
        if point2 == "null":
            return [paper_img, point1]
        else:
            return [img_on_paper, point2]
