import os
import sys

from kornia.enhance import normalize_min_max, AdjustGamma
from kornia.filters import laplacian

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image

from test_distortion import show_result
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
    print(net.residual.weight.shape)
    print(output)
    print(net)
    print(sum(p.numel() for p in net.parameters()))


def test_stn():
    """
    检测
    """
    img_path = "out/grayscale_trans.jpg"
    img = transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor()
    ])(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    # img = torch.randn((batch_size, 3, img_size, img_size)).to(device)

    net = STN().to(device)
    output = net(img)

    print(output.size())
    print(sum(p.numel() for p in net.parameters()))
    show_result(output)


def test_decoder_model():
    img = torch.randn((batch_size, 3, img_size, img_size)).to(device)
    msg = torch.randn((batch_size, msg_size)).to(device)

    net = MPDecoder(msg_size=msg_size, img_size=img_size, decoder_type="conv").to(device)
    output = net(img)

    print(output[0].size(), msg.size())
    msg_loss = F.binary_cross_entropy(output[0], msg)
    print(msg_loss)
    msg_loss.backward()
    print(sum(p.numel() for p in net.parameters()))

    for n, p in net.named_parameters():
        print(n)


def test_lpips():
    from lpips import LPIPS
    img_path = "test_source/lena.jpg"
    encoded_img_path = "../out/encoded.jpg"
    img = transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor()
    ])(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    encoded_img = transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor()
    ])(Image.open(encoded_img_path).convert("RGB")).unsqueeze(0).to(device)

    lpips_module = LPIPS(net="alex").to(device)
    lpips_loss = lpips_module(img, encoded_img).mean()
    print(lpips_loss)


def test_mask():
    img_path = "test_source/COCO_val2014_000000005037.jpg"
    img = transforms.Compose([
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor()
    ])(Image.open(img_path).convert("RGB")).to(device)
    mask = 1 - torch.abs(laplacian(img.unsqueeze(0), 3))  # low weight in high frequency
    mask = (AdjustGamma(40., 1.)(normalize_min_max(mask))).squeeze(0)
    mask = torch.round(mask)
    # show_result(img, None, True)
    show_result(mask, None, True)


if __name__ == '__main__':
    # test_encoder_model()
    # test_stn()
    # test_decoder_model()
    # test_lpips()
    test_mask()
