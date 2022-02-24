# encoding: utf-8
# 主函数

from flask import Flask, request
from flask_cors import CORS

import properties
from model.encoderLive import Live
from utils import *

app = Flask(__name__)

app.debug = True

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
    global custom_id, custom_info, is_live, live
    try:
        if (is_live == False):
            sql_id = custom_id if (custom_id is None) else properties.UID
            sql_info = custom_info if (custom_info is None) else properties.UIP
                
            live = Live(src=properties.SRC_RTMP_URL, dst=properties.DST_RTMP_URL, uid=sql_id, uip=sql_info)
            
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
        live.stop()
        is_live = False
        return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.STOP_STREAM_OK)
    except:
        return reponseJson(code=CodeEnum.HTTP_SERVER_ERROR, msg=MsgEnum.STOP_STREAM_FAIL)

# decoder 上传图片
@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    fileBase64 = data["fileBase64"]
    points = data["positions"]
    auto = data["auto"]
    imgType = data["type"]
    
    img = base64ToCv2Img(fileBase64)
    
    if (auto == 1):
        if (len(points) != 0):
            img = perspectiveTrans(img, points)
    # elif (auto == 2):
    #     img = autoPerspectiveTrans(img)
        
    # if (imgType == 1):
    #     # 若是截图，免 STN
    #     # decodedInfo = decoder(img)
    # elif (imgType is 2):
    #     # 若是照片，进 STN
    #     # decodedInfo = decoder_with_stn(img)
    
    # 解码后信息
    # decodedInfo = decoder(img)
    
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    fake_info = {"ts_min": 27384770, "uid": "123"}
    resList = selectLog(properties.SQLITE_LOCATION, fake_info["ts_min"], fake_info["uid"], 3)
    outDict = {"data": resList}
    
    # 返回变换后图像 base64
    outDict["fixedImg"] = cv2ImgToBase64(img)

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
    # GBlur = data["GBlur"]
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
    global custom_id, custom_info
    
    data = request.get_json()
    fileBase64 = data["fileBase64"]
    
    img = base64ToCv2Img(fileBase64)
    
    # 编码后图片
    # img = encoder(img)
    
    sql_id = custom_id if (custom_id is None) else properties.UID
    sql_info = custom_info if (custom_info is None) else properties.UIP
    
    insertLog(properties.SQLITE_LOCATION, getMinutesTs(), sql_id, sql_info)
    
    # 返回变换后图像 base64
    outDict = {"encodedImg": cv2ImgToBase64(img)}

    return reponseJson(code=CodeEnum.HTTP_OK, msg=MsgEnum.UPLOAD_OK, out_dict=outDict)

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, threaded = True)