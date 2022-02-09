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
              <el-button type="primary" @click="stopStreaming" style="margin-right: 15px;">停止推拉流</el-button>
              <el-button type="primary" @click="dialogVisible = true">上传图片</el-button>
            </el-form>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>

  <el-dialog
    id="dialog-container"
    class="dialog-class"
    title="上传图片并编码"
    :visible.sync="dialogVisible"
    :before-close="handleClose"
    width="70%"
    top="15vh">
    <div>
      <!-- 图片上传控件 -->
      <el-upload
        id="upload-unit"
        v-if="isShowUpload"
        drag
        action="#"
        list-type="picture"
        :auto-upload="false"
        :on-change="imgSaveToUrl"
        :accept="'image/*'">
        <i class="el-icon-upload" style="color:#409EFF"></i>
        <div class="el-upload__text text">
          将图片拖到此处, 或
          <em>点击上传</em>
        </div>
      </el-upload>

      <!-- 本地预览需要上传处理的图片 -->
      <div v-if="isShowImgUpload" id="preview">
        <div style="margin-bottom: 10px;">
          <el-button type="primary" :loading="false" size="small" @click="uploadButtonClick" style="margin-right: 10px;">重新上传</el-button>
          <el-button type="primary" :loading="false" size="small" @click="processButtonClick">确定上传</el-button>
        </div>
        <div>
          <el-image
            style="height: 200px; display: inline-block; margin-right: 20px;"
            :src="localUrl"
            :preview-src-list="[localUrl]"
            fit="scale-down">
          </el-image>
          <el-image
            style="height: 200px; display: inline-block;"
            :src="encodedUrl"
            :preview-src-list="[encodedUrl]"
            fit="scale-down">
            <div slot="error" class="image-slot">
              <i class="el-icon-picture-outline"></i>
            </div>
          </el-image>
        </div>
      </div>
    </div>
    <div slot="footer">
      <el-button type="primary" @click="handleClose">关闭</el-button>
    </div>
  </el-dialog>
</div>
</template>

<script>
const Player = null;
export default {
  name: "Encoder",
  data() {
    return {
      isShowUpload: true,
      isShowImgUpload: false,

      dialogVisible: false,
      
      localUrl: null,
      encodedUrl: null,

      form: {
        id: null,
        fileBase64: null
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
    },
    // 图片转 blob(url) 再转为 base64
    imgSaveToUrl(event) {
      // 获取上传图片的本地URL，用于上传前的本地预览
      let URL = null;
      if (window.createObjectURL != undefined) {
        // basic
        URL = window.createObjectURL(event.raw);
      } else if (window.URL != undefined) {
        // mozilla(firefox)
        URL = window.URL.createObjectURL(event.raw);
      } else if (window.webkitURL != undefined) {
        // webkit or chrome
        URL = window.webkitURL.createObjectURL(event.raw);
      }
      // 转换后的地址为 blob:http://xxx/7bf54338-74bb-47b9-9a7f-7a7093c716b5
      this.localUrl = URL;

      // 异步调用转换函数
      this.$func.file2Base64(event.raw).then(res => {
        this.form.fileBase64 = res.split(',')[1];
      }).catch(err => {
        console.log(err);
      })
      
      this.isShowImgUpload = true;  // 呈现本地预览组件
      this.isShowUpload = false;    // 隐藏上传组件
    },
    // 重新上传
    uploadButtonClick() {
      this.isShowImgUpload = false;
      this.isShowUpload = true;
      this.localUrl = null;
      this.encodedUrl = null;
    },
    // 上传图片
    processButtonClick() {
      const params = {
        fileBase64: this.form.fileBase64,
      };
      this.$http.uploadPicEn(params).then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });
          this.encodedUrl = "data:image/jpeg;base64," + data.encodedImg;
        }
      }).catch(err => {
        this.$message({
          message: '上传失败!',
          type: 'error',
          showClose: true
        }); 
        console.log(err);
      });
    },
    // 关闭前询问
    handleClose(done) {
      this.$confirm('确认关闭?')
        .then(_ => {
          this.dialogVisible = false;
          this.uploadButtonClick();
          done();
        })
        .catch(_ => {});
    },
  },
  mounted() {
    this.initPlayer();
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

#dialog-container {
  width: calc(100% - 400px);
  margin-left: 200px;
  height: calc(100vh - 142px);
  text-align: center;

  #upload-unit { 
    text-align: center;
  }

  #preview {
    text-align: center;
  }
}
</style>

<style scoped>
body /deep/ .el-card__body {
  padding-left: 0px;
  padding-right: 0px;
  padding-top: 30px;
  padding-bottom: 10px;
}

.dialog-class /deep/ .el-dialog__body {
  padding-top: 10px;
  padding-bottom: 10px;
}
</style>