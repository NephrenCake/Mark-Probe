# -- coding: utf-8 --
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse

from PIL import Image

from tools.interface.utils import get_device, model_import
from tools.interface.bch import BCHHelper
from tools.interface.predict import decode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='path of the image file (.png or .jpg)',
                        default="out/encoded.jpg")
    parser.add_argument('--model_path', help='path of the model file (.pth)',
                        default="weight/latest_4-8_seem_the_balanced.pth")
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    return parser.parse_args()


def main(args):
    img = Image.open(args.img_path)
    bch = BCHHelper()
    device = get_device(args.device)
    decoder = model_import(args.model_path, "Decoder", device=device)

    # 调用 api
    uid, time, content, msg_pred, score, bf = decode(img=img,
                                                     bch=bch,
                                                     device=device,
                                                     model=decoder,
                                                     use_stn=True)

    print("水印指向用户: ", uid)
    print("水印指向时间: ", time)
    print("水印原生内容: ", content)
    # print("水印正确率: ", )
    print("水印置信度: ", score)
    print("校验码纠正位数: ", bf)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
