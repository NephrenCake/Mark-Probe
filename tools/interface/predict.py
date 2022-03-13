from typing import Union, List

import cv2
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
           ) -> List[List]:
    assert target in ["screen", "paper"], "暂时只支持检测 screen 或 paper 上的隐写图像"
    if target == "screen":
        contour_img, point = predict(img, model, thresold_value=thresold_1)
        point1 = point[0][0]
        point2 = point[1][0]
        point3 = point[2][0]
        point4 = point[3][0]
        '''
            返回一个列表，列表里有四个点的坐标一个一个标好轮廓的图片
            假设返回一个列表名为res的列表，要取四个点则写成res[0],res[1],res[2],res[3]
            要取标好轮廓的图片则写成res[4]
        '''

        return [point1, point2, point3, point4, contour_img]
        # 返回标注好点的图片以及四个点的坐标,取四个点的坐标时可写为point[0][0],point[0][1],point[0][2],point[0][3]
    else:
        '''
        1.当照片场景较大时，需要分别进行两次检测才能完成，第一次检测是检测打印纸所在位置，第二此是检测打印纸上的图片
        第一次检测返回角度矫正好的打印纸图片和它在原图上的坐标，第二次检测返回打印纸上角度矫正好的图片和它在原图上的坐标
        2.当照片拍摄距离很近时，那么一次检测就够，返回角度矫正好的图片和它在原图上的坐标
        
        '''
        point, paper, contour_img, = paper_detect.paper_find1(img)
        point1 = point[0][0]
        point2 = point[1][0]
        point3 = point[2][0]
        point4 = point[3][0]
        '''
            返回一个列表，列表里有四个点的坐标一个一个标好轮廓的图片
            假设返回一个列表名为res的列表，要取四个点则写成res[0],res[1],res[2],res[3]
            要取标好轮廓的图片则写成res[4]
        '''

        return [point1, point2, point3, point4, contour_img]
