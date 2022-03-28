# encoding: utf-8
# 主函数

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from flask import Flask, request
from flask_cors import CORS

import service.backend.src.properties as properties
from service.backend.src.model.coder import encoder, decoder
from service.backend.src.model.encoderLive import Live
from service.backend.src.utils import *

from tools.interface.bch import BCHHelper
from tools.interface.predict import detect
from tools.interface.utils import model_import, get_device

from detection.Monitor_detection.deeplab import DeeplabV3

app = Flask(__name__)

# debug 模式会占用两倍显存：初始化了两次模型
app.debug = False

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db.sqlite3"

# 解决跨域
CORS(app)

# 数据库
# from flask_sqlalchemy import SQLAlchemy
# from werkzeug.utils import secure_filename
# db = SQLAlchemy(app)
# class Records(db.Model):
#     __tablename__ = 'records'
    
#     time_stamp = db.Column(db.Integer, primary_key=True, unique=True)
#     uid = db.Column(db.String(128), index=True)
#     ip = db.Column(db.String(128))
    
#     def __init__(self, ts, uid, ip):
#         self.time_stamp = ts
#         self.uid = uid
#         self.ip = ip
    
#     def __repr__(self):
#         return '<Records {},{},{}>'.format(str(self.time_stamp), self.uid, self.ip)

# 全局变量
custom_id = None
custom_info = None
live = None
is_live = False

# 模型引入
bch = BCHHelper()
device = get_device(properties.DEVICE)
encoderModel = model_import(model_path=properties.CODER_WEIGHT_PATH, model_name="Encoder", device=device, warmup=True)
decoderModel = model_import(model_path=properties.CODER_WEIGHT_PATH, model_name="Decoder", device=device, warmup=True)

# 检测引入
detectModel = DeeplabV3(properties.DETECT_WEIGHT_PATH)

# API
# 自定义 ID
@app.route("/custom", methods = ["POST"])
def customID():
    global custom_id, custom_info
    try:
        data = request.get_json()
        custom_id = data["id"]
        custom_info = data["extendInfo"]
        
        out_d = {"id": custom_id, "extendInfo": custom_info}
        
        if (custom_info != ""):
            s_msg = "设置成功! 当前 ID 为 " + custom_id + " 。当前自定义信息为 " + custom_info + " 。"
        else:
            s_msg = "设置成功! 当前 ID 为 " + custom_id + " 。当前未自定义信息。"
        
        return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.ID_SET_OK, out_dict=out_d, specify_msg=s_msg)
    except:
        return reponseJson(code=CodeEnum.HTTP_BAD_REQUEST, msg=MsgEnum.FAIL)

# 推流（包含拉流过程，先拉后推）
@app.route("/pull", methods = ["GET"])
def pullStream():
    global custom_id, custom_info, is_live, live, encoderModel, bch, device
    try:
        if (is_live == False):
            sql_id = properties.UID if custom_id is None else custom_id
            sql_info = properties.UIP if custom_info is None else custom_info
                
            live = Live(src=properties.SRC_RTMP_URL, dst=properties.DST_RTMP_URL, uid=sql_id, uip=sql_info, model=encoderModel, bch=bch, device=device)
            
            is_live = True
            live.run()
            return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.PULL_PUSH_STREAM_OK)
        else:
            return reponseJson(code=CodeEnum.HTTP_FORBBIDEN, msg=MsgEnum.UNDER_STREAMING)
    except:
        return reponseJson(code=CodeEnum.HTTP_SERVER_ERROR, msg=MsgEnum.PULL_PUSH_STREAM_FAIL)

# 停止推拉流
@app.route("/stop", methods = ["GET"])
def stopStream():
    global is_live, live
    try:
        is_live = False
        live.stop()
        return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.STOP_STREAM_OK)
    except:
        return reponseJson(code=CodeEnum.HTTP_SERVER_ERROR, msg=MsgEnum.STOP_STREAM_FAIL)

# decoder 上传图片
@app.route('/upload', methods=['POST'])
def upload():
    global decoderModel, bch, device, detectModel
    
    data = request.get_json()
    fileBase64 = data["fileBase64"]
    points = data["positions"]
    print(points)
    auto = data["auto"]
    imgType = data["type"]
    
    img = base64ToCv2Img(fileBase64)
    
    decodedInfo = {}
    
    try:
        if (auto == 1):
            if (len(points) != 0):
                img = perspectiveTrans(img, points, 1)
        elif (auto == 2):
            target = 'screen' if imgType == 1 else 'paper'
            detect_res = detect(img=img, model=detectModel, target=target, thresold_1=38)
            
            print(detect_res)
            
            # 检测给出的点的顺序是 正 Z 字形，从左上角开始！
            # detect_points = [
            #     {'id': 1, 'x': detect_res["1"], 'y': detect_res[0][0][0][1]},
            #     {'id': 2, 'x': detect_res[0][2][0][0], 'y': detect_res[0][2][0][1]},
            #     {'id': 3, 'x': detect_res[0][3][0][0], 'y': detect_res[0][3][0][1]},
            #     {'id': 4, 'x': detect_res[0][1][0][0], 'y': detect_res[0][1][0][1]}
            # ]
            img = perspectiveTrans(img, detect_res, 2)
    except:
        return reponseJson(code=CodeEnum.HTTP_SERVER_ERROR, msg=MsgEnum.PERSPECTIVETRANS_FAILURE)
    
        
    if (imgType == 1):
        # 若是截图，免 STN
        decodedInfo = decoder(img=img, use_stn=False, model=decoderModel, bch=bch, device=device)
    elif (imgType == 2):
        # 若是照片，进 STN
        decodedInfo = decoder(img=img, use_stn=True, model=decoderModel, bch=bch, device=device)
    
    # fake_info = {"ts_min": 27384770, "uid": "123"}
    resList = selectLog(properties.SQLITE_LOCATION, decodedInfo["ts"] / 60, decodedInfo["uid"], 0)
    outDict = {"data": resList}
    
    # 返回变换后图像 base64
    outDict["fixedImg"] = cv2ImgToBase64(img)
    
    # 返回解码得到的原信息和置信度
    outDict["content"] = decodedInfo["content"]
    outDict["score"] = decodedInfo["score"]

    return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.UPLOAD_OK, out_dict=outDict)

# ps 上传图片
@app.route('/ps', methods=['POST'])
def ps():
    data = request.get_json()
    fileBase64 = data["fileBase64"]
    
    # brightness = data["brightness"]
    # contrast = data["contrast"]
    # saturation = data["saturation"]
    # hue = data["hue"]
    # MBlur = data["MBlur"]
    # randomNoise = data["randomNoise"]
    # grayscale = data["grayscale"]
    # randomCover = data["randomCover"]
    # JpegZip = data["JpegZip"]
    
    img = base64ToCv2Img(fileBase64)
    img = psPic(img, data)

    # 返回变换后图像 base64
    outDict = {"fixedImg": cv2ImgToBase64(img)}

    return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.UPLOAD_OK, out_dict=outDict)

# encoder 上传图片
@app.route('/uploaden', methods=['POST'])
def uploadEn():
    global custom_id, custom_info, encoderModel, bch, device
    
    data = request.get_json()
    fileBase64 = data["fileBase64"]
    
    img = base64ToCv2Img(fileBase64)
    
    # 编码
    sql_id = properties.UID if custom_id is None else custom_id
    img = encoder(img=img, info=sql_id, model=encoderModel, bch=bch, device=device)
    
    # 写数据库
    # sql_id = properties.UID if custom_id is None else custom_id
    sql_info = properties.UIP if custom_info is None else custom_info
    insertLog(properties.SQLITE_LOCATION, getMinutesTs(), sql_id, sql_info)
    
    # 返回变换后图像 base64
    outDict = {"encodedImg": cv2ImgToBase64(img)}

    return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.UPLOAD_OK, out_dict=outDict)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=False)