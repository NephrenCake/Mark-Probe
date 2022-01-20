# encoding: utf-8
# 工具类与枚举类

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
    
    PULL_PUSH_STREAM_OK = "推拉流成功!"
    UNDER_STREAMING = "已经在推拉流!"
    PULL_PUSH_STREAM_FAIL = "推拉流失败!"
    STOP_STREAM_OK = "停止推拉流成功!"
    STOP_STREAM_FAIL = "停止推拉流失败!"
    
    UPLOAD_OK = "上传成功!"
    
    
# 格式化消息
def reponseJson(code, msg, out_dict = None):
    res_dict = {"code": code.value, "msg": msg.value}
    
    if (out_dict):
        res_dict.update(out_dict)
    
    return jsonify(res_dict)