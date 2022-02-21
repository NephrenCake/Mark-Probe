# encoding: utf-8
# 拉流，处理，推流

import queue
import threading
import cv2 as cv
import subprocess as sp

import properties
import utils

class Live(object):
    def __init__(self, src:str, dst:str, uid:str, uip:str):
        self.frame_queue = queue.Queue()
        self.command = ""
        
        # 管道
        self.pipe = None
        # 捕获
        self.cap = None

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

    # 读取流
    def read_frame(self):
        self.cap = cv.VideoCapture(self.camera_path)

        # 获取 帧率，长，宽
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # FFmpeg 指令
        self.command = ['ffmpeg',
                '-hwaccel_output_format', 'cuda',   # 使用 CUDA
                '-re',              # 输出速度与帧率一致
                '-r', str(fps),     # 帧率
                '-y',               # 覆盖视频
                '-an',              # 去音频
                '-f', 'rawvideo',   # 输入格式
                '-vcodec', 'rawvideo',   # 与 -c:v 等价，这里是输入的参数
                '-pix_fmt', 'bgr24',    # 输入颜色空间
                '-s', "{}x{}".format(width, height),    # 画面分辨率
                '-i', '-',              # 输入
                '-max_delay', '100',    # 最大延迟毫秒数
                '-g', '10',             # GOP 大小
                '-b:v', '500K',         # 视频码率（音频码率为 -b:a）
                '-c:v', 'h264_nvenc',   # 编解码器，libx264：H264-CPU，h264_nvenc：H264-CUDA
                '-pix_fmt', 'yuv420p',  # 输出颜色空间
                '-bufsize', '500K',     # 缓冲区大小，一般而言，码率*帧率*5
                # '-maxrate', '1000K'     # 最大码率
                # '-preset', 'ultrafast',      # 预设（CUDA 下不可用）
                '-f', 'flv',            # 输出格式
                self.rtmpUrl]           # 输出到的 URL
        
        # 读取流（可以是 rtmp 流，也可以是 摄像头 等等）
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                print("读取流失败!")
                break

            # 将 帧 放入队列
            try:
                self.frame_queue.put(frame)
            except:
                raise Exception("帧 进入队列失败!")

    # 推出流
    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                try:
                    self.pipe = sp.Popen(self.command, stdin=sp.PIPE)
                    break
                except:
                    raise Exception("FFmpeg 启动失败!") 
                
        while True:
            if self.frame_queue.empty() != True:
                # 从队列获取帧
                frame = self.frame_queue.get()
            
                # 处理逻辑
                # 这里预留动态帧率
                # if (做差(self.last_src_frame, frame) < 阈值):
                #     frame = encoder(frame)
                #     self.last_encoded_frame = frame
                #     self.last_src_frame = frame
                # else:
                #     frame = self.last_encoded_frame
                
                # 写数据库
                # present_minute = utils.getMinutesTs()
                # if (present_minute - self.last_minute >= 1):
                #     utils.insertLog(properties.SQLITE_LOCATION, present_minute, self.uid, self.uip)
                #     self.last_minute = present_minute
                
                # 写入管道
                self.pipe.stdin.write(frame.tobytes())
                
    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        # 守护进程与 join 必须注释，不然 Flask 在处理推流请求时无法返回指定信息导致超时。
        # [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]
        # [thread.join() for thread in threads]
        
    def stop(self):
        # 这里总会报奇怪的错误：（但不影响运行逻辑。）
        # 1. self.pipe.stdin.write(frame.tobytes())
        #       ValueError: write to closed file
        # 2. ret, frame = self.cap.read()
        #       cv2.error: Unknown C++ exception from OpenCV code
        
        self.pipe.stdin.close()
        self.pipe.communicate()
        self.cap.release()
        self.pipe.terminate()