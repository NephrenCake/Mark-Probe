# -- coding: utf-8 --
import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from typing import Union

import torch
from PIL import Image
from torchvision.transforms import transforms

import torchvision.transforms.functional as F

from steganography.models.MPNet import MPEncoder
from tools.interface.bch import BCHHelper


def encode(img: torch.Tensor, uid: Union[int, str], model: MPEncoder, bch: BCHHelper) -> (torch.Tensor, torch.Tensor):
    if len(img.size()) == 3:
        img = img.unsqueeze(0)

    dat, now, key = bch.convert_uid_to_data(uid)
    packet = bch.encode_data(dat)  # 数据段+校验段

    img_low = F.resize(img, [448, 448])
    res_low = model({"img": img_low, "msg": packet})
    res_high = F.resize(res_low, img.shape[-2:])
    encoded_img = torch.clamp(res_high + img, 0., 1.)

    return encoded_img.squeeze(0), res_low.squeeze(0)


def main():
    img_path = "data/val2014/COCO_val2014_000000000042.jpg"
    save_dir = "./out"
    pretrained = "train_log/justCrop_2021-10-29-10-51-17/latest-5.pth"

    img_size = (448, 448)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = MPEncoder().to(device)
    checkpoint = torch.load(pretrained, map_location=device)
    encoder.load_state_dict(checkpoint['Encoder'])
    encoder.eval()

    img = Image.open(img_path).convert("RGB")
    img = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])(img).to(device)

    bch = BCHHelper()

    encoded_img, res_img = encode(img=img, uid=114514, model=encoder, bch=bch)  # 调用

    encoded_img_save = transforms.ToPILImage()(encoded_img)
    res_img_save = transforms.ToPILImage()(res_img + 0.5)
    encoded_img_save.save(os.path.join(save_dir, "encoded_" + img_path.split("/")[-1]))
    res_img_save.save(os.path.join(save_dir, "residual_" + img_path.split("/")[-1]))


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
    main()
