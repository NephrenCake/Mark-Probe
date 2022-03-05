# -- coding: utf-8 --
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse
import torch
import numpy as np

from PIL import Image
from typing import Union
from torchvision.transforms import transforms

from steganography.models.MPNet import MPDecoder
from tools.interface.utils import get_device, convert_img_type
from tools.interface.bch import BCHHelper
from tools.encode import model_import


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

    return uid, time, content, msg_pred, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='path of the image file (.png or .jpg)',
                        default="out/encoded.jpg")
    parser.add_argument('--model_path', help='path of the model file (.pth)',
                        default="weight/latest-0.pth")
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    return parser.parse_args()


def main(args):
    img = Image.open(args.img_path)
    bch = BCHHelper()
    device = get_device(args.device)
    decoder = model_import(args.model_path, "Decoder", device=device)

    # 调用 api
    uid, time, content, msg_pred, score = decode(img=img,
                                             bch=bch,
                                             device=device,
                                             model=decoder,
                                             use_stn=True)

    print("水印指向用户: ", uid)
    print("水印指向时间: ", time)
    print("水印原生内容: ", content)
    # print("水印正确率: ", )
    print("水印置信度: ", score)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
