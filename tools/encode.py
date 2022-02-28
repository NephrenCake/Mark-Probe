# -- coding: utf-8 --
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from typing import Union

import torch
from PIL import Image
from torchvision.transforms import transforms

import torchvision.transforms.functional as F

from steganography.models.MPNet import MPEncoder
from tools.interface.bch import BCHHelper

# 这里配置默认的参数
img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\\test_source\COCO_test2014_000000000001.jpg"
model_path = "D:\learning\pythonProjects\HiddenWatermark1\steganography\\train_log\CI-test_2022-02-19-13-23-41\\best.pth"
output_path = "D:\learning\pythonProjects\HiddenWatermark1\\test/encode_output_file"
user_id = 114514
#



parser = argparse.ArgumentParser()
parser.add_argument('--img_path', help='path of the image file (.png or .jpg)', default=img_path)
parser.add_argument('--model_path', help='path of the model file (.pth)', default=model_path)
parser.add_argument('--output_path', help='folder path of the encoded images', default=output_path)
parser.add_argument('--user_id', help='the msg embedded in to the image',default=user_id)
args = parser.parse_args()


'''
encode:
       一张图片encode后 存放到指定的文件夹中
'''
def encode(img: np.ndarray, uid: Union[int, str], model: MPEncoder, bch: BCHHelper, device) -> (torch.Tensor, torch.Tensor):


    img_low = transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(device)

    img = transforms.ToTensor()(img)

    dat, now, key = bch.convert_uid_to_data(uid)
    packet =torch.tensor(bch.encode_data(dat), dtype=torch.float32).unsqueeze(0)   # 数据段+校验段

    res_low = model({"img": img_low, "msg": packet})
    res_high = F.resize(res_low, img.shape[-2:])
    encoded_img = torch.clamp(res_high + img, 0., 1.)

    return encoded_img.squeeze(0), res_low.squeeze(0)


def main():

    img_size = (448, 448)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    encoder = MPEncoder().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['Encoder'])
    encoder.eval()

    img = Image.open(args.img_path).convert("RGB")
    bch = BCHHelper()

    encoded_img, res_img = encode(img=img, uid=args.user_id, model=encoder, bch=bch, device=device)  # 调用

    encoded_img_save = transforms.ToPILImage()(encoded_img)
    res_img_save = transforms.ToPILImage()(res_img + 0.5)
    encoded_img_save.save(args.output_path+"/encoded.png")
    res_img_save.save(args.output_path+"/res.png")


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
    main()


