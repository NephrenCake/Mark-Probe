import argparse
import os
import random
import sys
import time
import numpy as np
import cv2
from PIL import Image

from tools.interface.bch import BCHHelper
from tools.interface.utils import get_device, model_import
from detection.Monitor_detection.deeplab import DeeplabV3
from detection.Paper_detection import utils
from tools.interface.predict import detect, decode

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_way', help='1.monitor 2.picture 3.monitor_video 4.picture_video',
                        default=4)
    parser.add_argument('--video_path', help='path of the video file',
                        default="")
    parser.add_argument('--video_save_path', help='folder path of the video',
                        default="")
    # ./ out / final_save_acc / 9.mp4
    parser.add_argument('--video_fps', help='the fps of save_video',
                        default=25)
    parser.add_argument('--paper_num', help='the number of paper',
                        default=2)
    parser.add_argument('--model_path', help='path of the model file (.pth)',
                        default="weight/latest-15.pth")
    parser.add_argument('--device', help='the model loaded in cpu(cpu) or gpu(cuda)',
                        default='cuda')

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
            frame_copy = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            res = detect(frame, deeplab, target="screen", thresold_1=55)  # res[0]为图片，res[1]为坐标

            if type(res) != int:

                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(res[4]['img'], cv2.COLOR_RGB2BGR)
            else:
                frame = frame_copy

            fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
        device = get_device(args.device)
        decoder = model_import(args.model_path, "Decoder", device=device)
        bch = BCHHelper()

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
        while True:

            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()

            if not ref:
                break
            frame_show = frame.copy()
            frame_warp = frame.copy()
            # 格式转变，BGRtoRGB

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = detect(frame, model=None, target="paper", num=3)
            # RGBtoBGR满足opencv显示格式
            if type(res) != int:
                for i in range(len(res)):
                    if type(res[i][0]) != int:


                        pts1 = np.float32(res[i][4]['all_point'])

                        pts2 = np.float32([[0, 0], [448, 0], [0, 448], [448, 448]])
                        matrix = cv2.getPerspectiveTransform(pts1, pts2)
                        imgWarpColored = cv2.warpPerspective(frame_warp, matrix, (448, 448))

                        uid, create_time, content, msg_pred, score, bf = decode(img=imgWarpColored,
                                                                                bch=bch,
                                                                                device=device,
                                                                                model=decoder,
                                                                                use_stn=True)

                        if True:
                            location = (res[i][0]['x'], res[i][0]['y'])
                            # character = str(114514) + ' ' + '2022-04-08 18:39:00'
                            # character = str(uid) + ' ' + create_time
                            # final_img = cv2.putText(final_img, character, location, cv2.FONT_HERSHEY_SIMPLEX,
                            #                         0.75,
                            #                         (0, 0, 0), 2)

                            score = int(score * 100)
                            # score = str(int(score * 100)) + '%'
                            if True:
                                area = cv2.contourArea(res[i][4]['all_point'])
                                print(area)
                                if area > 500:
                                    final_img = cv2.drawContours(frame_show, res[i][4]['all_point'], -1, (0, 255, 0),
                                                                 20)  # DRAW THE BIGGEST CONTOUR
                                    final_img = utils.drawRectangle(final_img, res[i][4]['all_point'], 2)

                                    # print(score)
                                    final_img = cv2.putText(final_img, str(score) + '%', location,
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.75,
                                                            (0, 0, 0), 2)
                                    frame_show = final_img
                        # print("水印指向用户: ", uid)
                        # print("水印指向时间: ", create_time)
                        # print("水印原生内容: ", content)
                    else:
                        pass

                frame = frame_show
                # frame = cv2.cvtColor(res[args.paper_num - 1][4]['img'], cv2.COLOR_RGB2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # fps = (fps + (1. / (time.time() - t1))) / 2
            # print("fps= %.2f" % (fps))
            # frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
