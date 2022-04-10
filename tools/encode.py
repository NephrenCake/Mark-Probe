# -- coding: utf-8 --
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse

from PIL import Image
from torchvision.transforms import transforms

from tools.interface.bch import BCHHelper
from tools.interface.utils import model_import, get_device, check_dir
from tools.interface.predict import encode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='path of the image file (.png or .jpg)',
                        default="test/test_source/lena.jpg")
    parser.add_argument('--model_path', help='path of the model file (.pth)',
                        default="weight/latest_4-8_seem_the_balanced.pth")
    parser.add_argument('--output_path', help='folder path of the encoded images',
                        default="out/")
    parser.add_argument('--user_id', help='the msg embedded in to the image',
                        default=114514)
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    return parser.parse_args()


def main(args):
    img = Image.open(args.img_path).convert("RGB")
    device = get_device(args.device)
    encoder = model_import(args.model_path, model_name="Encoder", device=device)
    bch = BCHHelper()

    # 调用 api
    encoded_img, res_img = encode(img0=img,
                                  uid=args.user_id,
                                  model=encoder,
                                  bch=bch,
                                  device=device,
                                  keep_size=False)

    check_dir(args.output_path)
    transforms.ToPILImage()(encoded_img).save(os.path.join(args.output_path, 'encoded.jpg'))
    transforms.ToPILImage()(res_img + 0.5).save(os.path.join(args.output_path, 'residual.jpg'))


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
