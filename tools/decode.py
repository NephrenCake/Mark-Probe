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


# 设置默认参数
img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\encode_output_file\encoded.png"
model_path = "D:\learning\pythonProjects\HiddenWatermark1\steganography\\train_log\CI-test_2022-02-19-13-23-41\\best.pth"


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', help='path of the image file (.png or .jpg)', default=img_path)
parser.add_argument('--model_path', help='path of the model file (.pth)', default=model_path)
args = parser.parse_args()

'''
这里的decode 的返回和bch解码msg_pre 的返回值一致。
'''
def decode(img:np.ndarray, bch:BCHHelper, model:MPDecoder, device, use_stn:bool) -> (int, str, str):
    img = transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor()
    ])(img).unsqueeze(0)
    msg_pred = torch.round(model(img, use_stn=use_stn)[0]).detach().numpy().ravel()
    bf, dat = bch.decode_data(msg_pred)
    return bch.convert_data_to_uid(bf, dat)



def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    img = Image.open(args.img_path).convert("RGB")
    bch = BCHHelper()
    decoder = MPDecoder(decoder_type="conv").to(device)
    net_state_dict = decoder.state_dict()
    checkpoint = torch.load(args.model_path, map_location=device)

    for k, v in net_state_dict.items():
        if k in checkpoint:
            net_state_dict[k] = checkpoint[k]
    decoder.load_state_dict(net_state_dict)
    decoder.eval()
    i_, now_, key_ = decode(img=img,bch=bch,device=device,model=decoder,use_stn=True)
    print(i_)
    print(now_)
    print(key_)


if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
    main()
