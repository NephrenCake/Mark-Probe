<template>
<div class="encoder-container">
  <div class="video-show">
    <el-row :gutter="0">
      <el-col :span="24" :offset="0">
        <el-card class="video-card">
          <div id="source-video">
            <p class="video-title">源视频流</p>
            <video class="videoElement" id="srcV" width="500px" height="350px" controls>本浏览器不支持 HTML5 视频, 请升级浏览器!</video>
            <el-button type="primary" class="video-btn" @click="pullStreamSource">{{srcVideo}}</el-button>
          </div>

          <div id="divider"></div>

          <div id="encoded-video">
            <p class="video-title">编码后视频流</p>
            <video class="videoElement" id="dstV" width="500px" height="350px" controls>本浏览器不支持 HTML5 视频, 请升级浏览器!</video>
            <el-button type="primary" class="video-btn" @click="pullStreamDst">{{dstVideo}}</el-button>
          </div>

          <div id="total-control-panel">
            <div id="present-time">当前时间: {{date_time}}</div>
            <el-form class="form-all" ref="form" :model="form" label-width="85px" :rules="rules">
              <el-form-item label="ID" prop="id">
                <el-input v-model="form.id" clearable style="width: 280px;"></el-input>
              </el-form-item>
              <el-form-item label="附加信息">
                <el-input type="textarea" v-model="form.extendInfo" style="width: 280px;"></el-input>
              </el-form-item>
            </el-form>

            <el-button type="primary" @click="postInfo">提交 ID 与 附加信息</el-button>
            <el-button type="primary" style="margin-left: 20px;" @click="streamControlAll">同时{{allVideo}}</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</div>
</template>

<script>
export default {
  name: "EncoderStream",
  data() {
    return {
      date_time: this.$func.getDateTime(),

      playerO: null,
      playerO_url: 'http://127.0.0.1:80/live?port=2935&app=live&stream=test',
      playerD: null,
      playerD_url: 'http://127.0.0.1:81/live?port=1935&app=live&stream=test',

      srcOn: false,
      dstOn: false,

      form: {
        id: null,
        extendInfo: null
      },

      rules: {
        id: [
          { required: true, message: '请填写 ID !', trigger: 'blur' }
        ]
      }
    }
  },
  methods: {
    // 编码信息设置
    postInfo() {
      if (!this.form.id) {
        this.$message({
          message: "请填写 ID !",
          type: 'warning',
          showClose: true
        });
        return;
      }
      const params = {
        id: this.form.id,
        extendInfo: this.form.extendInfo === null ? "" : this.form.extendInfo
      };
      this.$http.sendInfo(params).then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });
        }
      }).catch(err => {
        this.$message({
          message: '提交失败!',
          type: 'error',
          showClose: true
        }); 
        console.log(err);
      });
    },
    // 初始化播放器（务必有 return，否则因作用于问题无法赋值；下面的拉流和停止拉流同理）
    initPlayer(stream_url) {
      let player = this.$flv.createPlayer({
        type: "flv",
        hasAudio: false,
        isLive: true,
        fluid: true,
        stashInitialSize: 128,
        url: stream_url
      },{
        enableStashBuffer: false,
        fixAudioTimestampGap: false,
      });
      
      return player;
    },
    // 拉流操作函数
    startStreaming(stream_url, target_video_element_id) {
      let p = this.initPlayer(stream_url);
      let videoElement = document.getElementById(target_video_element_id);
      p.attachMediaElement(videoElement);
      p.load();
      p.play();

      return p;
    },
    // 停止拉流操作函数
    stopStreaming(player) {
      player.pause();
      player.unload();
      player.detachMediaElement();
      player.destroy();
      player = null;

      return player;
    },
    // 拉流控制：源视频流
    pullStreamSource() {
      if (!this.srcOn) {
        try {
          this.playerO = this.startStreaming(this.playerO_url, 'srcV');
          this.srcOn = true;
          this.$message({
            message: "源视频流拉取成功!",
            type: 'success',
            showClose: true
          });
        } catch(err) {
          console.log(err);
          this.$message({
            message: "源视频流拉取失败!",
            type: 'error',
            showClose: true
          });
        }
      } else {
        try {
          this.playerO = this.stopStreaming(this.playerO);
          this.srcOn = false;
          this.$message({
            message: "源视频流停止成功!",
            type: 'success',
            showClose: true
          });
        } catch(err) {
          console.log(err);
          this.$message({
            message: "源视频流停止失败!",
            type: 'error',
            showClose: true
          });
        }
      }
    },
    // 拉流控制：编码视频流
    pullStreamDst() {
      if (!this.dstOn) {
        this.$http.pullStream().then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });

          this.playerD = this.startStreaming(this.playerD_url, 'dstV');
          this.dstOn = true;

        } else if (data.code === 403) {
          this.$message({
            message: data.msg,
            type: 'warning',
            showClose: true
          });
        }
        }).catch(err => {
          this.$message({
            message: '编码视频流拉取失败!',
            type: 'error',
            showClose: true
          }); 
          console.log(err);
        });
      } else {
        this.$http.stopStream().then(res => {
          const data = res.data;
          if (data.code === 200) {
            this.$message({
              message: data.msg,
              type: 'success',
              showClose: true
            });
          } else if (data.code === 500) {
            this.$message({
              message: data.msg,
              type: 'warning',
              showClose: true
            });
          }
        }).catch(err => {
          this.$message({
            message: '编码视频流停止失败!',
            type: 'error',
            showClose: true
          }); 
          console.log(err);
        });

        this.playerD = this.stopStreaming(this.playerD);
        this.dstOn = false;
      }
    },
    // 拉流控制：二者
    streamControlAll() {
      if (!(this.srcOn && this.dstOn)) {
        if (!this.srcOn) {
          this.pullStreamSource();
        }
        if (!this.dstOn) {
          this.pullStreamDst();
        }
      } else {
        this.pullStreamSource();
        this.pullStreamDst();
      }
    } 
  },
  mounted() {
    this.timer = setInterval(() => {
      this.date_time = this.$func.getDateTime();
    }, 1000);
  },
  computed: {
    srcVideo() {
      return !this.srcOn ? '开始拉流' : '停止拉流';
    },
    dstVideo() {
      return !this.dstOn ? '开始拉流' : '停止拉流';
    },
    allVideo() {
      return !(this.srcOn && this.dstOn) ? '开始拉流' : '停止拉流';
    }
  },
  beforeDestroy() {
    // 在 Vue 实例销毁前，清除定时器
    if (this.timer) {
      clearInterval(this.timer);
    }
  }
};
</script>

<style lang="scss" scoped>
.video-title {
  margin: 50px 0 30px 0;
  font-size: 20px;
}
.video-btn {
  margin-top: 30px;
  margin-bottom: 10px;
}
.encoder-container {
  height: 100%;

  .video-card{
    text-align: center;

    #source-video {
      width: 500px;
      display: inline-block;
    }
    #encoded-video {
      width: 500px;
      display: inline-block;
    }
    #total-control-panel {
      margin-left: 20px;
      width: 370px;
      display: inline-block;

      #present-time {
        font-size: 20px;
      }
      .form-all {
        text-align: left;
        margin-top: 120px;
      }
    }
    #divider {
      height: 450px;
      width: 0px;
      display: inline-block;
      margin: 0 20px 0 21px;
      border-left: 2px solid #ccc;
    }
    
  }
}
.control-form {
  padding-top: 30px;
}
</style>