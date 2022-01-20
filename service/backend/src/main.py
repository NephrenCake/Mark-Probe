# encoding: utf-8
# 主函数

import properties
from model.encoder import Live
from flask import Flask, request
from utils import CodeEnum, MsgEnum, reponseJson

app = Flask(__name__)

# 全局变量
custom_id = None
live = Live(src = properties.SRC_RTMP_URL, dst = properties.DST_RTMP_URL)
is_live = False

# 自定义 ID
@app.route("/custom", methods = ["POST"])
def custom_id():
    global custom_id
    try:
        data = request.get_json()
        custom_id = data["id"]
        return reponseJson(code = CodeEnum.HTTP_OK, msg = MsgEnum.ID_SET_OK, out_dict = {"id": custom_id})
    except:
        return reponseJson(code = CodeEnum.HTTP_BAD_REQUEST, msg = MsgEnum.FAIL)

# 推流（包含拉流过程，先拉后推）
@app.route("/pull", methods = ["GET"])
def pullStream():
    global is_live, live
    try:
        if (is_live == False):
            is_live = True
            live.run()
            # 开始推流了就无法返回信息，可能多线程能够解决？（先搁置了；前端先做成：如果无返回信息就是成功）
            return reponseJson(code = CodeEnum.HTTP_OK, msg = MsgEnum.PULL_PUSH_STREAM_OK)
        else:
            return reponseJson(code = CodeEnum.HTTP_FORBBIDEN, msg = MsgEnum.UNDER_STREAMING)
    except:
        return reponseJson(code = CodeEnum.HTTP_SERVER_ERROR, msg = MsgEnum.PULL_PUSH_STREAM_FAIL)

@app.route("/stop", methods = ["GET"])
def stopStream():
    global is_live, live
    try:
        live.stop()
        is_live = False
        return reponseJson(code = CodeEnum.HTTP_OK, msg = MsgEnum.STOP_STREAM_OK)
    except:
        return reponseJson(code = CodeEnum.HTTP_SERVER_ERROR, msg = MsgEnum.STOP_STREAM_FAIL)


if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, threaded = True)