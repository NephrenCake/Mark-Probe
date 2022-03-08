# -- coding: utf-8 --
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from torch.backends import cudnn

sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import time
import cv2
import argparse

from tools.interface.bch import BCHHelper
from tools.interface.utils import model_import, get_device, check_dir, tensor_2_cvImage, convert_img_type
from tools.interface.predict import encode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path of the video file',
                        default="test/test_source/test.mp4")
    parser.add_argument('--decoder_model_path', help='path of the model file (.pth)',
                        default="weight/latest-0.pth")
    parser.add_argument('--detector_model_path', help='path of the model file (.pth)',
                        default="")
    parser.add_argument('--output_path', help='folder path of the encoded images',
                        default="out/")
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    return parser.parse_args()


# todo 注：该脚本暂时用于测试速度与优化探索！解码的视频脚本可以另开
def main(args):
    # check_dir(args.output_path)
    device = get_device(args.device)
    encoder = model_import(args.decoder_model_path, model_name="Encoder", device=device)
    bch = BCHHelper()

    # ==============
    cap = cv2.VideoCapture(args.video_path)
    counter = 0  # 设置一个counter 来计算平均帧率
    cudnn.benchmark = True  # 加快在视频中恒定大小图像的推断

    timer = 0.
    ret, img = cap.read()
    assert ret, "Can't receive any frame!"
    while ret and cv2.waitKey(1) != ord('q'):
        # 调用 api
        img = convert_img_type(img).to(device)
        t1 = time.time()
        encoded_img, _ = encode(img=img,
                                uid=114514,
                                model=encoder,
                                bch=bch,
                                device=device)
        counter += 1
        t2 = time.time()
        encoded_img = tensor_2_cvImage(encoded_img)

        timer += t2 - t1
        cv2.putText(img=encoded_img, text=f"FPS {float('%.2f' % (counter / timer))}", org=(50, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
        cv2.imshow('frame', encoded_img)
        ret, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
