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
                    <el-button type="primary" @click="((form.saturation = 100) && processButtonClick())">确定上传</el-button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="outer-container">
            <div id="divider"></div>
          </div>

          <div class="outer-container">
            <div id="ps-img">
              <p class="img-title">处理图</p>
              
              <div class="img-part">
                <el-image
                  style="width: 500px; height: 350px;"
                  :src="fixedUrl"
                  :preview-src-list="[fixedUrl]"
                  fit="scale-down">
                  <div slot="error" style="height: 350px; width: 500px; background: #eee;">
                    <i class="el-icon-picture-outline" style="padding-top: 150px; font-size: 40px; color: #888"></i>
                  </div>
                </el-image>

                <el-button type="primary" v-if="showDownload" class="img-btn" style="margin-right: 30px;">
                  <a :href="fixedUrl" download>下载处理图</a>
                </el-button>
                <el-button type="primary" v-if="showDownload" class="img-btn" @click="goDecoding">解码并溯源处理图</el-button>
              </div>
            </div>
          </div>

          <div class="outer-container">
            <div id="total-control-panel">

              <div class="slider-panel">
                <span class="slider-label">亮度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.brightness" :max="300" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.brightness"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">对比度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.contrast" :max="500" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.contrast"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">饱和度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.saturation" :max="100" @change="processButtonClick" :format-tooltip="formatValue_100" :disabled="notAllowOp" :marks="marks.saturation"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">色相&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.hue" :max="100" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.hue"></el-slider>
                </div>                
              </div>

              <div class="switch-panel">
                <span class="switch-label">高斯模糊&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="switch">
                  <el-switch v-model="form.GBlur" active-text="是" inactive-text="否" @change="processButtonClick" :disabled="notAllowOp"></el-switch>
                </div>                
              </div>

              <div class="slider-panel">
                <span class="slider-label">随机噪声&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.randomNoise" :max="200" @change="processButtonClick" :format-tooltip="formatValue_10K" :disabled="notAllowOp" :marks="marks.randomNoise"></el-slider>
                </div>                
              </div>

              <div class="switch-panel">
                <span class="switch-label" style="margin-right: 2px;">灰度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="switch">
                  <el-switch v-model="form.grayscale" active-text="是" inactive-text="否" @change="processButtonClick" :disabled="notAllowOp"></el-switch>
                </div>                
              </div>

              <div class="slider-panel">
                <span class="slider-label">随机遮挡&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.randomCover" :max="500" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.randomCover"></el-slider>
                </div>                
              </div>

              <div class="slider-panel">
                <span class="slider-label">Jpeg 压缩</span>
                <div class="slider">
                  <el-slider v-model="form.JpegZip" :max="5000" @change="processButtonClick" :format-tooltip="formatValue_100" :disabled="notAllowOp" :marks="marks.JpegZip"></el-slider>
                </div>                
              </div>
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
  name: "PS",
  data() {
    return {
      isShowUpload: true,
      isShowImgUpload: false,
      
      localUrl: null,
      fixedUrl: null,

      // 标记
      marks: {
        brightness: {
          0: '0',
          150: '0.15',
          300: '0.3'
        },
        contrast: {
          0: '0',
          250: '0.25',
          500: '0.5'
        },
        saturation: {
          0: '0',
          50: '0.5',
          100: '1'
        },
        hue: {
          0: '0',
          50: '0.05',
          100: '0.1'
        },
        randomNoise: {
          0: '0',
          100: '0.01',
          200: '0.02'
        },
        randomCover: {
          0: '0',
          250: '0.25',
          500: '0.5'
        },
        JpegZip: {
          0: '0',
          2500: '25',
          5000: '50'
        }
      },

      form: {
        // 亮度: 0-0.3
        brightness: null,
        // 对比度: 0-0.5
        contrast: null,
        // 饱和度: 0-1
        saturation: null,
        // 色相: 0-0.1
        hue: null,
        // 高斯模糊: true/false
        GBlur: null,
        // 随机噪声: 0-0.02
        randomNoise: null,
        // 灰度: true/false
        grayscale: null,
        // 随机遮挡: 0-0.5
        randomCover: null,
        // jpeg压缩: 0-50
        JpegZip: null,

        fileBase64: null
      }
    }
  },
  methods: {
    // 重新上传
    uploadButtonClick() {
      this.isShowImgUpload = false;
      this.isShowUpload = true;
      this.localUrl = null;
      this.fixedUrl = null;

      this.$func.clearObj(this.form);
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
        brightness: this.form.brightness / 1000,
        contrast: this.form.contrast / 1000,
        saturation: this.form.saturation / 100,
        hue: this.form.hue / 1000,
        GBlur: this.form.GBlur,
        randomNoise: this.form.randomNoise / 10000,
        grayscale: this.form.grayscale,
        randomCover: this.form.randomCover / 1000,
        JpegZip: this.form.JpegZip / 100,
        fileBase64: this.form.fileBase64
      };
      this.$http.psPic(params).then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });

          this.fixedUrl = this.$func.createDownloadFileUrl("PSPic.jpg", data.fixedImg);
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
    // 解码处理图
    goDecoding() {
      this.$store.commit('setPsPicToDecode', this.form.fileBase64);
      this.$router.push('/decoder/pic');
    },

    // 格式化处理 100 系数
    formatValue_100(val) {
      return (val / 100);
    },
    // 格式化处理 1000 系数
    formatValue_1K(val) {
      return (val / 1000);
    },
    // 格式化处理 10,000 系数
    formatValue_10K(val) {
      return (val / 1000 / 10).toFixed(4);
    }
  },
  mounted() {
  },
  computed: {
    // 显示下载按钮
    showDownload() {
      if (this.fixedUrl) {
        return true;
      } else {
        return false;
      }
    },
    notAllowOp() {
      return this.isShowUpload;
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
  #ps-img {
    width: 500px;
  }

  #total-control-panel {
    margin-top: 0px;
    margin-left: 20px;
    width: 370px;

    .switch-panel {
      height: 60px;

      .switch {
        display: inline-block;
        margin-top: 20px;
        width: 250px;
      }

      .switch-label {
        position: relative;
        right: 18px;
      }
    }

    .slider-panel {
      height: 60px;

      .slider-label {
        position: relative;
        bottom: 14px;
        right: 18px;
      }
      .slider {
        position: relative;
        left: 20px;
        margin-top: 11px;
        display: inline-block;
        width: 260px;
      }
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