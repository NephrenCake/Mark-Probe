import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import torch.nn.functional as F

from steganography.models.MPNet import MPEncoder, STN, MPDecoder

img_size = 448
msg_size = 96
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_encoder_model():
    img = torch.randn((batch_size, 3, img_size, img_size)).to(device)
    msg = torch.randn(batch_size, msg_size).to("cuda")

    net = MPEncoder().to("cuda")
    output = net({"img": img, "msg": msg})

    print(output.size())
    # print(net.residual.weight.shape)
    # print(output)
    # print(net)
    print(sum(p.numel() for p in net.parameters()))  # stega 1739999  MPEncoder 1713071


def test_stn():
    img = torch.randn((batch_size, 3, img_size, img_size)).to(device)

    net = STN().to(device)
    output = net(img)

    print(output.size())
    print(sum(p.numel() for p in net.parameters()))  # 41054150 -> 26079430


def test_decoder_model():
    img = torch.randn((batch_size, 3, img_size, img_size)).to(device)
    msg = torch.randn((batch_size, msg_size)).to(device)

    net = MPDecoder(msg_size=msg_size, img_size=img_size, decoder_type="conv").to(device)
    output = net(img)

    print(output[0].size(), msg.size())
    # msg_loss = F.binary_cross_entropy(output[0], msg)
    # print(msg_loss)
    # msg_loss.backward()
    print(sum(p.numel() for p in net.parameters()))  # stega 52505482  conv 39298182 swin 53672608


if __name__ == '__main__':
    test_encoder_model()
    test_stn()
    test_decoder_model()
