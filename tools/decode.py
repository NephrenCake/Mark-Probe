# -- coding: utf-8 --
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from encode import get_byte_msg
from steganography.models import stega_net
import numpy as np

from tools.utils.bch_utils import get_row_msg


def main():
    img_path = "out/encoded_COCO_val2014_000000000042.jpg"
    pretrained = "train_log/justCrop_2021-10-29-10-51-17/latest-5.pth"
    msg = "hello"

    msg_size = 100
    img_size = (400, 400)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    StegaStampDecoder = stega_net.StegaStampDecoder(msg_size).to(device)
    net_state_dict = StegaStampDecoder.state_dict()
    checkpoint = torch.load(pretrained, map_location=device)['Decoder']

    for k, v in net_state_dict.items():
        if k in checkpoint:
            net_state_dict[k] = checkpoint[k]
    StegaStampDecoder.load_state_dict(net_state_dict)
    StegaStampDecoder.eval()

    img = Image.open(img_path).convert("RGB")
    img = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])(img).to(device)

    msg_pred, _ = StegaStampDecoder(img.unsqueeze(0), use_stn=False)
    msg_pred = msg_pred.squeeze(0).cpu().detach()
    msg_pred = torch.round(msg_pred).numpy()

    msg_label = np.array(get_byte_msg(msg))

    wrong_num = np.count_nonzero(msg_label - msg_pred)
    print("bit_acc:", 1 - wrong_num / 100)

    get_row_msg(msg_pred)


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
    main()
