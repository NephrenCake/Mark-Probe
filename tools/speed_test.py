import datetime
import os
import threading
import time

import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from copy import deepcopy
import bchlib

from steganography.config.config import TrainConfig
from steganography.models import stega_net
import numpy as np

# from steganography.utils import TrainConfig
# from steganography.utils import make_trans, get_msg_acc

from steganography.utils.distortion import make_trans
from steganography.utils.train_utils import get_msg_acc

import cv2

BCH_POLYNOMIAL = 137
BCH_BITS = 5

thread_lock = threading.Lock()
thread_exit = False


# encode 部分
# 我需要修改encode函数使得其能直接读取图片
def encode(img_path, msg, save_dir, pretrained, image=None):
    '''
    img_path : 需要被encode的图片的位置
    msg:
    save_dir:
    pretrained: 训练模型
    image: picture
    中间两个不用我说都应该知道是啥了吧
    '''

    msg_size = 100
    img_size = (400, 400)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    StegaStampEncoder = stega_net.StegaStampEncoder(msg_size).to(device)
    checkpoint = torch.load(pretrained, map_location=device)
    StegaStampEncoder.load_state_dict(checkpoint['Encoder'])
    StegaStampEncoder.eval()

    # 这里存在着 PIL 到 cv2 图片形式的转换
    if img_path == None:
        # 这里面要将 opencv的图片转换为 pillow图片
        # img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = Image.open(img_path).convert("RGB")

    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
    ])(img).to(device)
    msg = np.array(get_byte_msg(msg))
    msg = torch.from_numpy(msg).to(torch.float).to(device)

    # 这里的res_img 是经过encode之后的残差图
    res_img = StegaStampEncoder({"img": img.unsqueeze(0), "msg": msg.unsqueeze(0)})
    encoded_img = img + res_img
    encoded_img = torch.clamp(encoded_img, 0., 1.)
    encoded_img_save = transforms.ToPILImage()(encoded_img.squeeze(0))

    encoded_img_save_cv = cv2.cvtColor(np.asarray(encoded_img_save), cv2.COLOR_RGB2BGR)

    return encoded_img_save_cv

    # if save_dir!= None:
    # # 将图片的 0--255 值 转换到 0--1 区间内
    #     encoded_img = torch.clamp(encoded_img, 0., 1.)
    #     torchvision.utils.save_image(encoded_img, os.path.join(save_dir, "encoded_" + img_path.split("/")[-1]))
    #
    #     encoded_img_save = transforms.ToPILImage()(encoded_img.squeeze(0))
    #     res_img_save = transforms.ToPILImage()(res_img.squeeze(0) + 0.5)
    #
    #     encoded_img_save.save(os.path.join(save_dir, "encoded_" + img_path.split("/")[-1]))
    #     res_img_save.save(os.path.join(save_dir, "residual_" + img_path.split("/")[-1]))
    #
    # pre_test = False
    # if pre_test:
    #     cfg = TrainConfig()
    #     cfg.set_iteration(10)
    #     cfg.setup_seed(2021)
    #     scales = cfg.get_cur_scales(cur_iter=0, cur_epoch=2)
    #     transformed_img = make_trans(encoded_img, scales)  # make_trans经常变动，因此此处经常需要修改
    #
    #     transformed_img_save = transforms.ToPILImage()(transformed_img.squeeze(0))
    #     transformed_img_save.save(os.path.join(save_dir, "transformed_" + img_path.split("/")[-1]))
    #
    #     # pre test
    #     transformed_img_save = Image.open(os.path.join(save_dir, "transformed_" + img_path.split("/")[-1])
    #                                       ).convert("RGB")
    #     transtrans_img = (transforms.ToTensor()(transformed_img_save)).to(device).unsqueeze(0)
    #     print(torch.nn.functional.mse_loss(transformed_img, transtrans_img))
    #     # pre test
    #     StegaStampDecoder = stega_net.StegaStampDecoder(msg_size).to(device)
    #     StegaStampDecoder.load_state_dict(checkpoint['Decoder'])
    #     StegaStampDecoder.eval()
    #
    #     msg_pred = StegaStampDecoder(transtrans_img)
    #     bit_acc, str_acc = get_msg_acc(msg, msg_pred)
    #     print(f"bit_acc: {bit_acc}, str_acc: {str_acc}")
    # return res_return


def get_byte_msg(msg):
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    if len(msg) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return
    # 补齐到7个字符，utf8编码
    data = bytearray(msg + ' ' * (7 - len(msg)), 'utf-8')
    ecc = bch.encode(data)  # bytearray(b'\x88\xa9\xfbN@')
    packet = data + ecc  # bytearray(b'Stega!!\x88\xa9\xfbN@')  12 = 7 + 5 字节
    # 校验码，两者加起来最多96bits
    packet_binary = ''.join(format(x, '08b') for x in packet)  # 转二进制
    # '010100110111010001100101011001110110000100100001001000011000100010101001111110110100111001000000'
    byte_msg = [int(x) for x in packet_binary]  # 转数组，len=96
    byte_msg.extend([0, 0, 0, 0])  # 补到len=100
    return byte_msg


# decode 部分
def get_row_msg(msg_pred):
    packet_binary = "".join([str(int(bit)) for bit in msg_pred[:96]])
    packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

    bitflips = bch.decode_inplace(data, ecc)
    if bitflips != -1:
        try:
            code = data.decode("utf-8")
            print(code)
            return
        except:
            return
    print('Failed to decode')


def decode(img_path, msg, pretrained):
    # img_path = "encode_output_file/encoded_COCO_val2014_000000000042.jpg"
    # pretrained = "train_log/test_CustomPer_2021-12-03-14-26-56/latest-5.pth"
    # msg = "hello"

    msg_size = 100
    img_size = (400, 400)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

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


# 定义一个处理视频中单帧的process
def process(frame, pretrained):
    res = encode(image=frame, img_path=None, pretrained=pretrained, save_dir=None)
    return res


class myThread(threading.Thread):
    def __init__(self, video_path, pretrained, msg):
        super(myThread, self).__init__()
        self.pretrained = pretrained
        self.msg = msg
        self.frame = np.zeros([400, 400, 3], np.uint8)
        self.video_path = video_path

    def get_frame(self):
        return deepcopy(self.frame)

    def run(self):
        global thread_exit
        cap = cv2.VideoCapture(self.video_path)
        while not thread_exit:
            ret, frame = cap.read()
            if ret:
                frame = encode(image=frame, img_path=None, msg=self.msg, save_dir=None, pretrained=self.pretrained)
                thread_lock.acquire()
                self.frame = frame
                thread_lock.release()
            else:
                thread_exit = True
        cap.release()


def singl_operation_runtime_consumption():
    # 读取一张图片
    pic_name = "COCO_val2014_000000000042.jpg"
    pre_path = "D:\learning\COCOTrain+Val/val2014"
    img_path = pre_path + "/" + pic_name
    msg = "hello"
    save_dir = "./encode_output_file"
    out_path = save_dir + "/" + "encoded_" + pic_name
    pretrained = "train_log/test_CustomPer_2021-12-03-14-26-56/latest-5.pth"

    start = datetime.datetime.now()
    encode(img_path=img_path, msg=msg, save_dir=save_dir, pretrained=pretrained)
    end_encode_time = datetime.datetime.now()
    decode(img_path=out_path, msg=msg, pretrained=pretrained)
    end_decode_time = datetime.datetime.now()

    print("encode time consumption:", end_encode_time - start)
    print("decode time consumption:", end_decode_time - end_encode_time)
    # 将encode 和 decode 的消耗时间返回
    return end_decode_time - start, end_decode_time - end_encode_time


def video_procession_consumption():  # 使用视频流输出
    pre_path = "D:\learning\COCOTrain+Val/theTestVideo"
    video_name = "001Video.mp4"
    video_path = pre_path + "/" + video_name
    pretrained = f"D:\learning\pythonProjects\HiddenWatermark-master/train_log/test_CustomPer_2021-12-03-14-26-56/latest-5.pth"
    # 视屏的读取：
    cap = cv2.VideoCapture(video_path)
    # start_time
    start_time = time.time()
    # 获得视屏的平均帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 设置一个counter 来计算平均帧率
    counter = 0
    # 获得视屏的原宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while (True):
        ret, frame = cap.read()
        # encode
        res = encode(image=frame, img_path=None, pretrained=pretrained, save_dir=None, msg="hello")
        # 由于 res 现在是tensor的格式！！！
        # res = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
        #
        cv2.imshow('frame', res)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
        # counter += 1
        # # if (time.time() - start_time) != 0:  # 实时显示帧数  这一步判断也是不必要的
        # cv2.putText(res, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
        #             3)
        # # src = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        # cv2.imshow('frame', res)
        # print("FPS: ", counter / (time.time() - start_time)) # 这一步不必要
        # counter = 0
        # start_time = time.time()
        # time.sleep(1 / fps)  # 按原帧率播放
    cap.release()
    cv2.destroyAllWindows()


# 多线程处理：

def video_thread_process():
    global thread_exit
    pre_path = "D:\learning\COCOTrain+Val/theTestVideo"
    video_name = "001Video.mp4"
    video_path = pre_path + "/" + video_name
    pretrained = f"D:\learning\pythonProjects\HiddenWatermark-master/train_log/test_CustomPer_2021-12-03-14-26-56/latest-5.pth"
    thread = myThread(video_path,pretrained,"hello")
    thread.start()

    while not thread_exit:
        thread_lock.acquire()
        frame = thread.get_frame()
        thread_lock.release()
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    thread.join()



if __name__ == "__main__":
    # singl_operation_runtime_consumption()
    video_procession_consumption()
    # video_thread_process()
