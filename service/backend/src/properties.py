# encoding: utf-8
# 设置

import os

# 来源 rtmp 流
SRC_RTMP_URL = "rtmp://127.0.0.1:2935/live/test"

# 目标 rtmp 流
DST_RTMP_URL = "rtmp://127.0.0.1:1935/live/test"

# ID
UID = "123"

# IP
UIP = "127.0.0.2"

# sqlite3 数据库位置
SQLITE_FILE_NAME = "db.sqlite3"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_LOCATION = os.path.join(BASE_DIR, SQLITE_FILE_NAME)

# 图像解码后，查询数据库时，设定的分钟误差
DELTA = 60