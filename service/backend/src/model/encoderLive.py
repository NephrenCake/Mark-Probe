# encoding: utf-8
# 拉流，处理，推流

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../../..')))

import time
import queue
import threading
import cv2
import subprocess as sp

from service.backend.src import properties, utils
from tools.interface.utils import tensor_2_cvImage, convert_img_type
from tools.interface.predict import encode

# counter = 0
# timer = .0

class Live(object):
    def __init__(self, src:str, dst:str, uid:str, uip:str, model, bch, device):
        # 源帧队列
        self.frame_queue = queue.Queue(maxsize=2)
        # 编码帧队列
        self.encoded_frame_queue = queue.Queue(maxsize=2)
        
        # FFmpeg 指令
        self.command = ""
        # 管道
        self.pipe = None
        # 捕获
        self.cap = None
        # 推流状态
        self.stats = True

        # 来源
        self.camera_path = src
        # 目标
        self.rtmpUrl = dst
        # ID
        self.uid = uid
        # IP
        self.uip = uip
        
        # 上一分钟时间戳
        self.last_minute = utils.getMinutesTs()
        # 上一源帧
        self.last_src_frame = None
        # 上一处理后帧
        self.last_encoded_frame = None
        
        # 实例化一个模型
        self.device = device
        self.bch = bch
        self.encoderModel = model
        
    # 读取流
    def read_frame(self):        
        self.cap = cv2.VideoCapture(self.camera_path)
        
        # 设置视频读取格式，直接转码为 Motion Jpg
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # 获取 帧率，长，宽
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FFmpeg 指令
        self.command = ['ffmpeg',
                '-hwaccel_output_format', 'cuda',   # 使用 CUDA
                '-re',              # 输出速度与目标帧率一致
                '-r', str(fps),     # 帧率
                '-y',               # 覆盖视频
                '-an',              # 去音频
                '-f', 'rawvideo',   # 输入格式
                '-vcodec', 'rawvideo',  # 与 -c:v 等价，这里是输入的参数
                '-pix_fmt', 'bgr24',    # 输入颜色空间：因为是 cv2 读取，所以是 bgr
                '-s', "{}x{}".format(width, height),    # 画面分辨率
                '-i', '-',              # 输入
                '-max_delay', '100',    # 最大延迟毫秒数
                '-g', '10',             # GOP 大小
                '-b:v', '3000K',         # 视频码率（音频码率为 -b:a）
                '-c:v', 'h264_nvenc',   # 编解码器，libx264：H264-CPU，h264_nvenc：H264-CUDA
                '-pix_fmt', 'yuv420p',  # 输出颜色空间
                '-bufsize', '5000K',     # 缓冲区大小，一般而言，码率*帧率*5
                # '-maxrate', '1000K'     # 最大码率
                # '-preset', 'ultrafast',      # 预设（CUDA 下不可用）
                '-f', 'flv',            # 输出格式
                self.rtmpUrl]           # 输出到的 URL

        # 读取流（可以是 rtmp 流，也可以是 摄像头 等等）
        while(self.cap.isOpened() and self.stats == True):
            ret, frame = self.cap.read()
            if not ret:
                print("读取流失败!")
                break

            # 将 原始帧 放入队列
            try:
                # frame = convert_img_type(frame).to(self.device)
                self.frame_queue.put(frame)
                
                # 剔除旧帧
                self.frame_queue.get() if self.frame_queue.qsize() > 1 else time.sleep(0.01)
            except:
                raise Exception("原始帧进入队列失败!")

    # 处理流
    def handle_frame(self):
        # global counter, timer

        while True:
            # 防止多线程时 command 未被设置
            if len(self.command) > 0:
                # 管道配置
                try:
                    self.pipe = sp.Popen(self.command, stdin=sp.PIPE)
                    break
                except:
                    raise Exception("FFmpeg 启动失败!")
                
        while(self.stats == True):
            if self.frame_queue.empty() != True:
                # 从队列获取原始帧
                frame = self.frame_queue.get()
            
                # 处理逻辑
                # 这里预留动态帧率
                # if (做差(self.last_src_frame, frame) < 阈值):
                #     frame = encoder(frame)
                #     self.last_encoded_frame = frame
                #     self.last_src_frame = frame
                # else:
                #     frame = self.last_encoded_frame
                
                frame = convert_img_type(frame).to(self.device)
                encoded_frame, res_frame = encode(img0=frame, uid=self.uid, model=self.encoderModel, bch=self.bch, device=self.device)
                encoded_frame = tensor_2_cvImage(encoded_frame)
                
                # 将 编码帧 放入队列
                # try:
                #     self.encoded_frame_queue.put(encoded_frame)
                #     self.encoded_frame_queue.get() if self.encoded_frame_queue.qsize() > 1 else time.sleep(0.01)
                # except:
                #     raise Exception("编码帧进入队列失败!")
                
                # 编码帧率测试
                # t1 = time.time()
                # counter += 1
                # t2 = time.time()
                # timer += t2 - t1
                # cv2.putText(img=encoded_frame, text=f"FPS {float('%.2f' % (counter / timer))}", org=(50, 50),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
                        
                # 写数据库
                present_minute = utils.getMinutesTs()
                if (present_minute - self.last_minute >= 1):
                    utils.insertLog(properties.SQLITE_LOCATION, present_minute, self.uid, self.uip)
                    self.last_minute = present_minute
                
                # 写入管道
                self.pipe.stdin.write(encoded_frame.tobytes())
                
    def push_frame(self):
        while True:
            # 防止多线程时 command 未被设置
            if len(self.command) > 0:
                # 管道配置
                try:
                    self.pipe = sp.Popen(self.command, stdin=sp.PIPE)
                    break
                except:
                    raise Exception("FFmpeg 启动失败!")
                
        while(self.stats == True):
            if self.encoded_frame_queue.empty() != True:
                encoded_frame = self.encoded_frame_queue.get()
                
                # encoded_frame = tensor_2_cvImage(encoded_frame)
                
                # # 写数据库
                # present_minute = utils.getMinutesTs()
                # if (present_minute - self.last_minute >= 1):
                #     utils.insertLog(properties.SQLITE_LOCATION, present_minute, self.uid, self.uip)
                #     self.last_minute = present_minute
                
                # 写入管道
                self.pipe.stdin.write(encoded_frame.tobytes())
    
    def run(self):
        self.stats = True
        read_frame_thread = threading.Thread(target=Live.read_frame, args=(self,))
        handle_frame_thread = threading.Thread(target=Live.handle_frame, args=(self,))
        # push_frame_thread = threading.Thread(target=Live.push_frame, args=(self,))
        
        # (self) 不是 tuple；(self,) 是 tuple
        # read_frame_process = mp.Process(target=self.read_frame, args=(self,))
        # handle_frame_process = mp.Process(target=self.handle_frame, args=(self,))
        # push_frame_process = mp.Process(target=self.push_frame, args=(self,))

        threads = [read_frame_thread, handle_frame_thread]
        # processes = [read_frame_process, handle_frame_process]
        
        # 守护进程与 join 必须注释，不然 Flask 在处理推流请求时无法返回指定信息导致超时。
        # [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]
        
        # [setattr(process, "daemon", True) for process in processes]
        # [process.start() for process in processes]
        # [process.join() for process in processes]
        
    def stop(self):
        self.stats = False
        # 务必等待几秒，从而让管道内数据全部处理完毕后实现正常退出
        time.sleep(1)
        self.cap.release()
        self.pipe.kill()