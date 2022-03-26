import argparse
import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.Monitor_detection.deeplab import DeeplabV3
from tools.interface.predict import detect


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_way', help='1.monitor 2.picture 3.monitor_video 4.picture_video',
                        default=3)
    parser.add_argument('--video_path', help='path of the video file',
                        default="test/test_source/test.mp4")
    parser.add_argument('--video_save_path', help='folder path of the video',
                        default="")
    parser.add_argument('--video_fps', help='the fps of save_video',
                        default=25)

    return parser.parse_args()


model_path = 'detection/Monitor_detection/logs/ep048-loss0.065-val_loss0.095.pth'


def main(args):
    if args.predict_way == 3:
        capture = cv2.VideoCapture(args.video_path)
        # 视频流保存
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.video_save_path, fourcc, args.video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        # 读取视频
        fps = 0.0
        deeplab = DeeplabV3(model_path)
        while True:
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            res = detect(frame, deeplab, target="screen", thresold_1=55)  # res[0]为图片，res[1]为坐标

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(res[4]['img'], cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if args.video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if args.video_save_path != "":
            print("Save processed video to the path :" + args.video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif args.predict_way == 4:  # 对打印纸检测

        capture = cv2.VideoCapture(args.video_path)
        # 视频流保存
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(args.video_save_path, fourcc, args.video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取视频，请注意是否正确填写视频路径。")
        # 读取视频
        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = detect(frame, model=None, target="paper")
            # RGBtoBGR满足opencv显示格式
            if res != -1:
                frame = cv2.cvtColor(res[4]['img'], cv2.COLOR_RGB2BGR)
            else:
                frame = frame
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video", frame)
            c = cv2.waitKey(33) & 0xff
            if args.video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if args.video_save_path != "":
            print("Save processed video to the path :" + args.video_save_path)
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main(parse_args())
