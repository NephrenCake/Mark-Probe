<template>
<div class="encoder-container">
  <div class="video-show">
    <el-row :gutter="0">
      <el-col :span="14" :offset="5">
        <el-card class="video-card">
          <video id="videoElement" width="800px" height="450px" controls>Your browser is too old which doesn't support HTML5 video.</video>
          <div class="control-form">
            <el-form ref="form" :model="form" label-width="50px" :rules="rules" inline>
              <el-form-item label="ID" prop="id" style="margin-right: 20px;">
                <el-input v-model="form.id" clearable></el-input>
              </el-form-item>
              <el-button type="primary" @click="postID" style="margin-right: 60px;">提交 ID</el-button>
              <el-button type="primary" @click="runStreaming" style="margin-right: 15px;">开始推拉流</el-button>
              <el-button type="primary" @click="stopStreaming">停止推拉流</el-button>
            </el-form>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</div>
</template>

<script>
const Player = null;
export default {
  name: "Encoder",
  data() {
    return {
      form: {
        id: null
      },

      rules: {
        id: [
          { required: true, message: '请填写 ID !', trigger: 'blur' }
        ]
      }
    }
  },
  methods: {
    // ID 设置
    postID() {
      const params = {
        id: this.form.id
      };
      this.$http.sendID(params).then(res => {
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
    // 开始推拉流
    runStreaming() {
      this.$http.pullStream().then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });

          this.initPlayer();
          let videoElement = document.getElementById('videoElement');
          this.Player.attachMediaElement(videoElement);
          this.Player.load();
          this.Player.play();

        } else if (data.code === 403) {
          this.$message({
            message: data.msg,
            type: 'warning',
            showClose: true
          });
        }
      }).catch(err => {
        this.$message({
          message: '推拉流失败!',
          type: 'error',
          showClose: true
        }); 
        console.log(err);
      });
    },
    // 停止推拉流
    stopStreaming() {
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
          message: '停止推拉流失败!',
          type: 'error',
          showClose: true
        }); 
        console.log(err);
      });

      this.Player.pause();
      this.Player.unload();
      this.Player.detachMediaElement();
      this.Player.destroy();
      this.Player = null;
    },
    // 初始化播放器
    initPlayer() {
      this.Player = this.$flv.createPlayer({
        type: "flv",
        hasAudio: false,
        isLive: true,
        fluid: true,
        stashInitialSize: 128,
        url: 'http://127.0.0.1/live?app=live&stream=test'
      },{
        enableStashBuffer: false,
        fixAudioTimestampGap: false,
        isLive: true
      });
    }
  },
  mounted() {
    this.initPlayer()
  }
};
</script>

<style lang="scss" scoped>
.encoder-container {
  height: 100%;

  .video-card{
    text-align: center;
  }
}
.control-form {
  padding-top: 30px;
}
</style>

<style scoped>
body /deep/ .el-card__body {
  padding-left: 0px;
  padding-right: 0px;
  padding-top: 30px;
  padding-bottom: 10px;
}
</style>