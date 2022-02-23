<template>
<div class="encoder-container">
  <div>
    <el-row :gutter="0">
      <el-col :span="24" :offset="0">
        <el-card class="img-card">
          <div class="outer-container">
            <div id="source-img">
              <p class="img-title">原图</p>
              
              <div class="img-part">
                <el-upload
                  style="padding-top: 70px;"
                  v-if="isShowUpload"
                  drag
                  action="#"
                  list-type="picture"
                  :auto-upload="false"
                  :on-change="imgSaveToUrl"
                  :accept="'image/*'">
                  <i class="el-icon-upload" style="color:#409EFF"></i>
                  <div class="el-upload__text text">
                    将图片拖到此处, 或<em>点击上传</em>
                  </div>
                </el-upload>
                
                <div v-if="isShowImgUpload" id="preview">
                  <div>
                    <el-image
                      style="width: 500px; height: 350px;"
                      :src="localUrl"
                      :preview-src-list="[localUrl]"
                      fit="scale-down">
                    </el-image>
                  </div>

                  <div class="img-btn">
                    <el-button type="primary" @click="uploadButtonClick" style="margin-right: 30px;">重新上传</el-button>
                    <el-button type="primary" @click="processButtonClick">确定上传</el-button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="outer-container">
            <div id="divider"></div>
          </div>

          <div class="outer-container">
            <div id="encoded-img">
              <p class="img-title">编码图</p>
              
              <div class="img-part">
                <el-image
                  style="width: 500px; height: 350px;"
                  :src="encodedUrl"
                  :preview-src-list="[encodedUrl]"
                  fit="scale-down">
                  <div slot="error" style="height: 350px; width: 500px; background: #eee;">
                    <i class="el-icon-picture-outline" style="padding-top: 150px; font-size: 40px; color: #888"></i>
                  </div>
                </el-image>

                <el-button type="primary" v-if="showDownload" class="img-btn">
                  <a :href="encodedUrl" download>下载编码图</a>
                </el-button>
              </div>
            </div>
          </div>

          <div class="outer-container">
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
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</div>
</template>

<script>
export default {
  name: "EncoderPic",
  data() {
    return {
      date_time: this.$func.getDateTime(),

      isShowUpload: true,
      isShowImgUpload: false,
      
      localUrl: null,
      encodedUrl: null,

      form: {
        id: null,
        extendInfo: null,
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
    // 重新上传
    uploadButtonClick() {
      this.isShowImgUpload = false;
      this.isShowUpload = true;
      this.localUrl = null;
      this.encodedUrl = null;
      this.form.fileBase64 = null;
    },
    // 图片转 blob(url) 再转为 base64
    imgSaveToUrl(event) {
      this.localUrl = this.$func.img2Blob(event);

      // 异步调用转换函数
      this.$func.file2Base64(event.raw).then(res => {
        this.form.fileBase64 = res.split(',')[1];
      }).catch(err => {
        console.log(err);
      })

      this.isShowImgUpload = true;
      this.isShowUpload = false;
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

          this.encodedUrl = this.$func.createDownloadFileUrl("encodedPic.jpg", data.encodedImg);
        }
      }).catch(err => {
        this.$message({
          message: '上传失败!',
          type: 'error',
          showClose: true
        }); 
        console.log(err);
      });
    }
  },
  mounted() {
    this.timer = setInterval(() => {
      this.date_time = this.$func.getDateTime();
    }, 1000);
  },
  computed: {
    // 显示下载按钮
    showDownload() {
      if (this.encodedUrl) {
        return true;
      } else {
        return false;
      }
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
.outer-container {
  display: inline-block;
  text-align: center;
  vertical-align: top;
  height: 536.2px;

  #source-img {
    width: 500px;
    margin-left: 11px;
  }
  #encoded-img {
    width: 500px;
  }

  #total-control-panel {
    margin-top: 205.2px;
    margin-left: 20px;
    width: 370px;

    #present-time {
      font-size: 20px;
    }
    .form-all {
      text-align: left;
      margin-top: 120px;
    }
  }
}

.img-title {
  margin: 50px 0 30px 0;
  font-size: 20px;
}
.img-part {
  height: 350px;
}
#divider {
  height: 450px;
  width: 0px;
  margin: 61px 20px 0 21px;
  border-left: 2px solid #ccc;
}
.control-form {
  padding-top: 30px;
}
.img-btn {
  margin-top: 30px;
  margin-bottom: 10px;
}
</style>