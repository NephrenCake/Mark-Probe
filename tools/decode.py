# -- coding: utf-8 --
import argparse
import os
import sys


sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from PIL import Image
from torchvision import transforms

from tools.interface.bch import BCHHelper
from steganography.models import stega_net
import numpy as np
from steganography.models.MPNet import MPDecoder
from tools.encode import model_import


# 设置默认参数
img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\encode_output_file\encoded.png"
model_path = "D:\learning\pythonProjects\HiddenWatermark1\steganography\\train_log\CI-test_2022-03-03-14-42-06\latest-0.pth"
device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', help='path of the image file (.png or .jpg)', default=img_path)
parser.add_argument('--model_path', help='path of the model file (.pth)', default=model_path)
parser.add_argument('--device',help='the model loaded in cpu(cpu) or gpu(cuda)',default=device)
args = parser.parse_args()

'''
这里的decode 的返回和bch解码msg_pre 的返回值一致。
'''
def decode(img, bch:BCHHelper, model:MPDecoder, device, use_stn:bool) -> (int, str, str, np.ndarray):
    img = transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor()
    ])(img).unsqueeze(0)
    msg_pred = torch.round(model(img, use_stn=use_stn)[0]).detach().numpy().ravel()


    bf, dat = bch.decode_data(msg_pred)
    i_, now_, key_ = bch.convert_data_to_uid(bf, dat)
    return i_, now_, key_, msg_pred



def main():
    img = Image.open(args.img_path)
    bch = BCHHelper()
    decoder = model_import(args.model_path,"Decoder",_device=torch.device(args.device))
    i_, now_, key_ ,msg_pred= decode(img=img,bch=bch,device=device,model=decoder,use_stn=False)
    print(i_)
    print(now_)
    print(key_)


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
    main()
