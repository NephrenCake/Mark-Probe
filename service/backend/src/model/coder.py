# encoding: utf-8
# 编码器与解码器

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../../..')))

import numpy as np

from service.backend.src import properties
from service.backend.src.utils import str_to_timestamp
from tools.interface.bch import BCHHelper
from tools.interface.utils import model_import, get_device, tensor_2_cvImage, convert_img_type
from tools.interface.predict import encode, decode

device = get_device(properties.DEVICE)
bch = BCHHelper()

# 编码器
def encoder(img: np.ndarray, info: str) -> np.ndarray:
    global device, bch
    
    img = convert_img_type(img).to(device)
    encoder = model_import(model_path=properties.WEIGHT_PATH, model_name="Encoder", device=device)

    encoded_img, res_img = encode(img=img, uid=info, model=encoder, bch=bch, device=device)
    
    encoded_img = tensor_2_cvImage(encoded_img)
    
    return encoded_img


# 解码器
def decoder(img: np.ndarray, use_stn: bool) -> dict:
    global device, bch
    decoder = model_import(model_path=properties.WEIGHT_PATH, model_name="Decoder", device=device)
    
    uid, time, content, msg_pred, score, bf = decode(img=img, bch=bch, model=decoder, device=device, use_stn=use_stn)
    
    # print("水印指向用户: ", uid)
    # print("水印指向时间: ", time) # time 为格式化时间 yyyy-MM-dd HH:mm:ss
    # print("水印原生内容: ", content)
    # print("水印正确率: ", msg_pred)
    # print("水印置信度: ", score)
    # print("水印纠正位: ", bf)
    
    return {"ts": str_to_timestamp(time), "uid": uid, "content": content, "score": score}