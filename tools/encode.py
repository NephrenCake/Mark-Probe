# -- coding: utf-8 --
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms.functional as F

from typing import Union
from PIL import Image
from torchvision.transforms import transforms
from steganography.models.MPNet import MPEncoder

from tools.interface.bch import BCHHelper
from tools.interface.utils import model_import, get_device


def encode(img: Union[np.ndarray, Image.Image, torch.Tensor],
           uid: Union[int, str],
           model: MPEncoder,
           bch: BCHHelper,
           device) -> (torch.Tensor, torch.Tensor):
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
    img = img.to(device)

    img_low = transforms.Resize(model.img_size)(img)
    dat, now, key = bch.convert_uid_to_data(uid)
    packet = torch.tensor(bch.encode_data(dat), dtype=torch.float32, device=device).unsqueeze(0)  # 数据段+校验段

    res_low = model({"img": img_low, "msg": packet})
    res_high = F.resize(res_low, img.shape[-2:])
    encoded_img = torch.clamp(res_high + img, 0., 1.)

    return encoded_img.squeeze(0), res_low.squeeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='path of the image file (.png or .jpg)',
                        default="test/test_source/COCO_train2014_000000000009.jpg")
    parser.add_argument('--model_path', help='path of the model file (.pth)',
                        default="weight/latest-0.pth")
    parser.add_argument('--output_path', help='folder path of the encoded images',
                        default="out/")
    parser.add_argument('--user_id', help='the msg embedded in to the image',
                        default=114514)
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    return parser.parse_args()


def main(args):
    img = Image.open(args.img_path).convert("RGB")
    device = get_device(args.device)
    encoder = model_import(args.model_path, model_name="Encoder", device=device)
    bch = BCHHelper()

    # 调用 api
    encoded_img, res_img = encode(img=img,
                                  uid=args.user_id,
                                  model=encoder,
                                  bch=bch,
                                  device=device)

    transforms.ToPILImage()(encoded_img).save(os.path.join(args.output_path, 'encoded.jpg'))
    transforms.ToPILImage()(res_img + 0.5).save(os.path.join(args.output_path, 'residual.jpg'))


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
