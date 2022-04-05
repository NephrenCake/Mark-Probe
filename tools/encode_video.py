# -- coding: utf-8 --
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + os.sep + '..')

import time
import cv2
import argparse
import torch

from torch.backends import cudnn

from tools.interface.bch import BCHHelper
from tools.interface.utils import model_import, get_device, tensor_2_cvImage, convert_img_type, check_dir
from tools.interface.predict import encode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='path of the video file',
                        default="test/test_source/test.mp4")
    parser.add_argument('--decoder_model_path', help='path of the model file (.pth)',
                        default="weight/latest-0.pth")
    parser.add_argument('--detector_model_path', help='path of the model file (.pth)',
                        default="")
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')
    parser.add_argument('--show_FPS', help='show FPS in the upper left corner',
                        default=False)
    parser.add_argument('--video_save_path', help='folder path of the video file',
                        default="out/encoded_video.mp4")
    return parser.parse_args()


# todo 注：该脚本暂时用于测试速度与优化探索！解码的视频脚本可以另开
def main(args):
    # check_dir(args.video_save_path)
    device = get_device(args.device)
    encoder = model_import(args.decoder_model_path, model_name="Encoder", device=device)
    bch = BCHHelper()

    packet = torch.tensor(bch.encode_data(bch.convert_uid_to_data(114514)[0]),
                          dtype=torch.float32, device=device).unsqueeze(0)

    # ==============
    cap = cv2.VideoCapture(args.video_path)
    counter = 0  # 设置一个counter 来计算平均帧率
    cudnn.benchmark = True  # 加快在视频中恒定大小图像的推断

    writer = cv2.VideoWriter(args.video_save_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    timer = 0.
    ret, img = cap.read()
    assert ret, "Can't receive any frame!"
    while ret and cv2.waitKey(1) != ord('q'):
        # 调用 api
        t1 = time.time()
        img = convert_img_type(img).to(device)
        encoded_img, _ = encode(img=img,
                                uid=114514,
                                model=encoder,
                                bch=bch,
                                device=device,
                                direct_msg=packet)
        counter += 1
        encoded_img = tensor_2_cvImage(encoded_img)
        t2 = time.time()

        timer += t2 - t1
        if args.show_FPS:
            cv2.putText(img=encoded_img, text=f"FPS {float('%.2f' % (counter / timer))}", org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
        cv2.imshow('frame', encoded_img)

        writer.write(encoded_img)
        ret, img = cap.read()

    if args.video_save_path != "":
        print("Save processed video to the path :" + args.video_save_path)
        writer.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
