# -- coding: utf-8 --
import os
import sys

import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms

from steganography.config.config import TrainConfig
from steganography.models import stega_net
import numpy as np

from tools.utils.bch_utils import get_byte_msg
from steganography.utils.distortion import make_trans
from steganography.utils.train_utils import get_msg_acc


def main():
    img_path = "data/val2014/COCO_val2014_000000000042.jpg"
    msg = "hello"
    save_dir = "./out"
    pretrained = "train_log/justCrop_2021-10-29-10-51-17/latest-5.pth"

    msg_size = 100
    img_size = (400, 400)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    StegaStampEncoder = stega_net.StegaStampEncoder(msg_size).to(device)
    checkpoint = torch.load(pretrained, map_location=device)
    StegaStampEncoder.load_state_dict(checkpoint['Encoder'])
    StegaStampEncoder.eval()

    img = Image.open(img_path).convert("RGB")
    img = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])(img).to(device)
    msg = np.array(get_byte_msg(msg))
    msg = torch.from_numpy(msg).to(torch.float).to(device)

    res_img = StegaStampEncoder({"img": img.unsqueeze(0), "msg": msg.unsqueeze(0)})
    encoded_img = img + res_img
    encoded_img = torch.clamp(encoded_img, 0., 1.)

    torchvision.utils.save_image(encoded_img, os.path.join(save_dir, "encoded_" + img_path.split("/")[-1]))

    encoded_img_save = transforms.ToPILImage()(encoded_img.squeeze(0))
    res_img_save = transforms.ToPILImage()(res_img.squeeze(0) + 0.5)

    encoded_img_save.save(os.path.join(save_dir, "encoded_" + img_path.split("/")[-1]))
    res_img_save.save(os.path.join(save_dir, "residual_" + img_path.split("/")[-1]))

    pre_test = False
    if pre_test:
        cfg = TrainConfig()
        cfg.set_iteration(10)
        cfg.setup_seed(2021)
        scales = cfg.get_cur_scales(cur_iter=0, cur_epoch=2)
        transformed_img = make_trans(encoded_img, scales)  # make_trans经常变动，因此此处经常需要修改

        transformed_img_save = transforms.ToPILImage()(transformed_img.squeeze(0))
        transformed_img_save.save(os.path.join(save_dir, "transformed_" + img_path.split("/")[-1]))

        # pre test
        transformed_img_save = Image.open(os.path.join(save_dir, "transformed_" + img_path.split("/")[-1])
                                          ).convert("RGB")
        transtrans_img = (transforms.ToTensor()(transformed_img_save)).to(device).unsqueeze(0)
        print(torch.nn.functional.mse_loss(transformed_img, transtrans_img))
        # pre test
        StegaStampDecoder = stega_net.StegaStampDecoder(msg_size).to(device)
        StegaStampDecoder.load_state_dict(checkpoint['Decoder'])
        StegaStampDecoder.eval()

        msg_pred = StegaStampDecoder(transtrans_img)
        bit_acc, str_acc = get_msg_acc(msg, msg_pred)
        print(f"bit_acc: {bit_acc}, str_acc: {str_acc}")


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.remove(__dir__)
    sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
    main()
