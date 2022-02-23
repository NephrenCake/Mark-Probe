# encoding: utf-8
# 工具类与枚举类

import cv2
import base64
import time
import sqlite3
import numpy as np
from enum import Enum
from flask import jsonify

# HTTP 状态枚举类
class CodeEnum(Enum):
    HTTP_OK = 200
    HTTP_BAD_REQUEST = 400
    HTTP_FORBBIDEN = 403
    HTTP_NOT_FOUND = 404
    HTTP_SERVER_ERROR = 500

# 消息枚举类
class MsgEnum(Enum):
    OK = "请求成功!"
    FAIL = "请求失败!"
    NOT_FOUND = "找不到资源!"
    BAD_REQUEST = "请求错误!"
    
    ID_SET_OK = "ID 设置成功!"
    
    PULL_STREAM_OK = "拉流成功!"
    PUSH_STREAM_OK = "推流成功!"
    
    PULL_PUSH_STREAM_OK = "编码视频流拉取成功!"
    UNDER_STREAMING = "已经在推拉流!"
    PULL_PUSH_STREAM_FAIL = "编码视频流拉取失败!"
    STOP_STREAM_OK = "编码视频流停止成功!"
    STOP_STREAM_FAIL = "编码视频流停止失败!"
    
    UPLOAD_OK = "上传成功!"

# 格式化消息
def reponseJson(code, msg, out_dict = None, specify_msg = None):
    res_dict = {"code": code.value, "msg": msg.value}
    
    if (out_dict):
        res_dict.update(out_dict)
        
    if (specify_msg):
        res_dict["msg"] = specify_msg
    
    return jsonify(res_dict)

# 获取分钟级时间戳
def getMinutesTs():
    return int(time.time() / 60)

# sqlite 原生操作数据库：插入数据
def insertLog(db:str, timeStamp:int, uid:str, ip:str):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    
    sql = "INSERT INTO RECORDS (TIME_STAMP,UID,IP) \
        VALUES ({}, '{}', '{}')".format(timeStamp, uid, ip)
    
    c.execute(sql)
    conn.commit()
    conn.close()
    
# sqlite 原生操作数据库：查询数据。delta 为时间误差允许范围，单位为分钟
def selectLog(db:str, timeStamp:int, uid:str, delta:int) -> list:
    conn = sqlite3.connect(db)
    c = conn.cursor()
    
    sql = "SELECT * FROM RECORDS WHERE UID='{}' AND TIME_STAMP>={} AND TIME_STAMP<={}".format(uid, timeStamp - delta, timeStamp + delta)
    
    cursor = c.execute(sql).fetchall()
    
    # 结果集
    listDict = []
    
    for row in cursor:
        # 分钟级时间戳转格式化时间
        ts = int(row[0] * 60)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        
        temp = {"id": row[1], "timeStamp": dt, "ip": row[2]}
        listDict.append(temp)
    
    conn.close()
    
    return listDict

# base64 转 cv2
def base64ToCv2Img(baseStr:str):
    imgString = base64.b64decode(baseStr)
    nparr = np.fromstring(imgString, np.uint8)  
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 返回的是一个 cv2 图片对象
    return image

# cv2 转 base64
def cv2ImgToBase64(image) -> str:
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str).decode()
    return base64_str

# CV2 透视变换（坐标顺序：↖，↙，↘，↗（逆时针顺序））
def perspectiveTrans(img, ratioPos:list):
    width = img.shape[1]
    height = img.shape[0]
    
    # 强制指定图片变换到 400 × 400
    # width = height = 400
    
    for ele in ratioPos:
        ele["x"] = (ele["x"] * width)
        ele["y"] = (ele["y"] * height)
    
    # 变换前的四个角点坐标
    former = np.float32([[ratioPos[0]["x"], ratioPos[0]["y"]],
                         [ratioPos[1]["x"], ratioPos[1]["y"]],
                         [ratioPos[2]["x"], ratioPos[2]["y"]],
                         [ratioPos[3]["x"], ratioPos[3]["y"]]
                         ])
    # 变换之后的四个角点坐标
    pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
 
    # 变换矩阵 M
    M = cv2.getPerspectiveTransform(former, pts)
    res = cv2.warpPerspective(img, M, (width, height))

    return res
    