# encoding: utf-8
# 拉流，处理，推流
# 处理帧的核心逻辑在第 77 行部分

import queue
import threading
import cv2 as cv
import subprocess as sp

class Live(object):
    def __init__(self, src, dst):
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

    def read_frame(self):
        # print("开启推流")
        self.cap = cv.VideoCapture(self.camera_path)

        # Get video information
        fps = int(self.cap.get(cv.CAP_PROP_FPS))
        width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # ffmpeg command
        self.command = ['ffmpeg',
                '-r', '30',
                '-y',
                '-an',              # 去音频
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                # '-max_delay', '100',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-i', '-',
                '-g', '30',         # GOP 大小
                '-b', '700000',     # 缓存大小
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv', 
                self.rtmpUrl]
        
        # read webcamera
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            # put frame into queue
            self.frame_queue.put(frame)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                self.pipe = sp.Popen(self.command, stdin=sp.PIPE)
                break

        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                # process frame
                
                # 处理逻辑
                
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                # cv.imshow('', frame)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                
                
                # write to pipe
                self.pipe.stdin.write(frame.tobytes())
                
    def run(self):
        threads = [
            threading.Thread(target = Live.read_frame, args = (self,)),
            threading.Thread(target = Live.push_frame, args = (self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        
    def stop(self):
        self.cap.release()
        self.pipe.terminate()