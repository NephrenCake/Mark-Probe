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

                <el-button type="primary" v-if="showDownload" class="img-btn" style="margin-right: 30px;" @click="resetPic">重置参数</el-button>
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
                  <el-slider v-model="form.brightness" :min="700" :max="1300" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.brightness"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">对比度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.contrast" :min="500" :max="1500" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.contrast"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">饱和度&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.saturation" :max="200" @change="processButtonClick" :format-tooltip="formatValue_100" :disabled="notAllowOp" :marks="marks.saturation"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">色相&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.hue" :max="200" @change="processButtonClick" :format-tooltip="formatValue_1K_Negative" :disabled="notAllowOp" :marks="marks.hue"></el-slider>
                </div>
              </div>

              <div class="slider-panel">
                <span class="slider-label">运动模糊&nbsp;&nbsp;</span>
                <div class="slider">
                  <el-slider v-model="form.MBlur" :max="3" :min="0" @change="processButtonClick" :disabled="notAllowOp" :marks="marks.MBlur"></el-slider>
                </div>                
              </div>

              <!-- <div class="switch-panel">
                <span class="switch-label">高斯模糊&nbsp;&nbsp;&nbsp;&nbsp;</span>
                <div class="switch">
                  <el-switch v-model="form.MBlur" active-text="是" inactive-text="否" @change="processButtonClick" :disabled="notAllowOp"></el-switch>
                </div>                
              </div> -->

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
                  <el-slider v-model="form.randomCover" :max="100" @change="processButtonClick" :format-tooltip="formatValue_1K" :disabled="notAllowOp" :marks="marks.randomCover"></el-slider>
                </div>                
              </div>

              <div class="slider-panel">
                <span class="slider-label">Jpeg 压缩</span>
                <div class="slider">
                  <el-slider v-model="form.JpegZip" :max="99" @change="processButtonClick" :disabled="notAllowOp" :marks="marks.JpegZip"></el-slider>
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

      fixedImgBase64: null,

      // 标记
      marks: {
        brightness: {
          700: '0.7',
          1000: '1.0',
          1300: '1.3'
        },
        contrast: {
          500: '0.5',
          1000: '1.0',
          1500: '1.5'
        },
        saturation: {
          0: '0',
          100: '1',
          200: '2'
        },
        hue: {
          0: '-0.1',
          100: '0',
          200: '0.1'
        },
        MBlur: {
          0: '0',
          1: '1',
          2: '2',
          3: '3'
        },
        randomNoise: {
          0: '0',
          100: '0.01',
          200: '0.02'
        },
        randomCover: {
          0: '0',
          50: '0.05',
          100: '0.1'
        },
        JpegZip: {
          0: '0',
          50: '50',
          99: '99'
        }
      },

      form: {
        // 亮度: 0.5-1.5, default: 1
        brightness: 1000,
        // 对比度: 0.7-1.3, default: 1
        contrast: 1000,
        // 饱和度: 0-2, default: 1
        saturation: 100,
        // 色相: -0.1-0.1, default: 0
        hue: 100,
        // 运动模糊: 0 (disabled), 1, 2, 3, default: 0
        MBlur: 0,
        // 随机噪声: 0-0.02, default: 0
        randomNoise: 0,
        // 灰度: true/false, default: false
        grayscale: false,
        // 随机遮挡: 0-0.1, default: 0
        randomCover: 0,
        // jpeg压缩: 0-99, default: 0
        JpegZip: 0,

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
      this.fixedImgBase64 = null;

      this.$func.clearObj(this.form);

      this.resetPs();
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
        hue: this.form.hue / 1000 - 0.1,
        MBlur: this.form.MBlur,
        randomNoise: this.form.randomNoise / 10000,
        grayscale: this.form.grayscale,
        randomCover: this.form.randomCover / 1000,
        JpegZip: this.form.JpegZip
      };
      this.$http.psPic(params).then(res => {
        const data = res.data;
        if (data.code === 200) {
          this.$message({
            message: data.msg,
            type: 'success',
            showClose: true
          });

          this.fixedImgBase64 = data.fixedImg;
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
      this.$store.commit('setPsPicToDecode', this.fixedImgBase64);
      this.$router.push('/decoder/pic');
    },
    // 重置图像攻击参数
    resetPs() {
      this.form.brightness = 1000;
      this.form.contrast = 1000;
      this.form.saturation = 100;
      this.form.hue = 100;
      this.form.MBlur = 0;
      this.form.randomNoise = 0;
      this.form.grayscale = false;
      this.form.randomCover = 0;
      this.form.JpegZip = 0;
    },
    // 重置图像
    resetPic() {
      this.resetPs();
      this.processButtonClick();
    },

    // 格式化处理 100 系数
    formatValue_100(val) {
      return (val / 100);
    },
    // 格式化处理 1000 系数
    formatValue_1K(val) {
      return (val / 1000);
    },
    // 格式化处理 1000 系数, 含负数
    formatValue_1K_Negative(val) {
      return ((val / 1000) - 0.1).toFixed(3); 
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