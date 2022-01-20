# 服务端实现

V 0.1.20220120

### 架构

> 目前仅就 视频流 的 Encoder 而言

1. 首先，第三方软件进行推流。
2. 服务端拿到第三方流，对流的每帧进行计算（自然是多线程的处理与计算）：
   - 首先，计算是不是与上一帧近似（近似算法是什么？阈值又该如何设置？）（这里会保存上一帧的 `base64`）
   - 再次，如果不近似，则将**该帧送入** Encoder 进行编码（同时送入一个自定义的 **7** 位的 `Char` 用于隐写编码），Encoder 应返回**残差图与原图相加完成的结果图**。然后，将该处理完毕的帧的 `base64` 编码存入当前线程的一个全局变量中。如若近似，跳过 Encoder ，直接将上一帧的处理后的图作为结果图送入视频流。
   - 接着，完成编码后，在另一新线程中操作数据库，将当前用户的 ID、IP、分钟级时间戳等信息存入数据库。
   - 最后，进行推流。（这里需要注意了：H.264 编码 以及 码率 会影响图像传输时导致的高频信息丢失；这里需要设置一个合适的码率，码率越高丢失越少，但是对应的视频流的延迟会增加。）
3. 客户端拿到视频流，前端显示视频流画面。



### 实现了什么？

> 由于 RTMP 协议的延迟问题，这里摒弃了 Java。

 - **第三方推流**
   - [x] 采用 OBS 进行推流（存在 1s 左右延迟；之后考虑直接命令行推流，优化推流参数来降低延迟）。
   
 - **服务端（采用 Flask 框架）**
   
   - [x] 自定义 ID 接口。
   - [x] 拉流：第三方视频流。
   - [ ] 动态帧率的判定。
   - [ ] 帧处理。（需要训练好的 Encoder 模型）
   - [ ] 持久化。
   - [x] 推流：`FFmpeg` 推流。
   
 - **客户端（采用 Vue.js）**
   
   - [ ] `flv` 播放器。
   - [ ] ID 修改框与逻辑。
   - [ ] 布局与样式设计。
   
 - **流媒体服务器（Nginx + nginx-http-flv-module）**
   - [x] rtmp 协议的设置。
   
   
### 目录说明

- `backend`：

  - `src`：

    - `model`：存放训练完毕的模型，并暴露为一个接口，形如：

      ```python
      # frame 为处理的帧，encoder 为编码器模型
      frame = encoder(frame, **kwargs)
      ```

    - `utils.py`：工具与枚举类。

    - `properties.py`：配置类。

    - `main.py`：主函数类。

- `frontend`：

  - `public`：入口网页，仅 `index.html` 需要修改。

  - `src`：
    - `assets`：静态资源。
    - `components`：组件。
    - `router`：路由。
    - `store`：Vuex（本项目不使用）。
    - `views`：视图。
    - `App.vue`：入口文件。
    - `main.js`：配置文件。



### 如何部署？

1. FFmpeg

   - 在 www.ffmpeg.org 下载编译完成的 Windows 版本的即可。
   - 配置环境变量。

2. Nginx

   Nginx 在本机上运行为两个实例，一个用来做 OBS 的推流服务，另一个用来做 服务端 的推流服务。

   这里提供一个编译完成的带有 flv 模块的版本，需要修改的就是 `Nginx` 根目录下 `conf` 文件夹下的 `nginx.conf` 中：

   ```bash
       server {
           listen 1935; # 运行端口，一个是 1935，另一个是 2935
   
           chunk_size 4000;
           application live {
               live on;
   			gop_cache on;
   			hls on;
               hls_path D:/nginx-rtmp/html/hls; # 这里改为 Nginx 对应 绝对路径 下的 html/hls，两个 Nginx 都要改
   ```

   - 双击运行 `nginx.exe` 即可作为后台服务启动。

3. Python 环境

   ```bash
   pip install flask opencv-python
   ```

4. 运行步骤：

   - 启动两个 Nginx；

   - 启动第三方推流；

   - 启动 `main.py` ；

   - 向 `127.0.0.1:5000` 发送请求：

     ```bash
     修改自定义 ID（json 传输）：
     POST: 127.0.0.1:5000/custom
     
     {
         "id": "123"
     }
     
     开始推拉流：
     GET: 127.0.0.1:5000/pull
     
     结束推拉流：
     GET: 127.0.0.1:5000/stop
     ```

     