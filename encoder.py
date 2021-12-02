# -- coding: utf-8 --
import os

import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
import bchlib

from steganography.models import stega_net
import numpy as np

from steganography.utils import TrainConfig
from steganography.utils import make_trans, get_msg_acc

BCH_POLYNOMIAL = 137
BCH_BITS = 5


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


def get_byte_msg(msg):
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    if len(msg) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return
    # 补齐到7个字符，utf8编码
    data = bytearray(msg + ' ' * (7 - len(msg)), 'utf-8')
    ecc = bch.encode(data)  # bytearray(b'\x88\xa9\xfbN@')
    packet = data + ecc  # bytearray(b'Stega!!\x88\xa9\xfbN@')  12 = 7 + 5 字节
    # 校验码，两者加起来最多96bits
    packet_binary = ''.join(format(x, '08b') for x in packet)  # 转二进制
    # '010100110111010001100101011001110110000100100001001000011000100010101001111110110100111001000000'
    byte_msg = [int(x) for x in packet_binary]  # 转数组，len=96
    byte_msg.extend([0, 0, 0, 0])  # 补到len=100
    return byte_msg


if __name__ == '__main__':
    main()
