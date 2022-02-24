<template>
<div class="encoder-container">
  <div>
    <el-row :gutter="0">
      <el-col :span="24" :offset="0">
        <el-card class="img-card">
          <div class="outer-container">
            <div id="source-img">
              <p class="img-title">泄漏图</p>
              
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
                    <el-button type="primary" @click="uploadButtonClick">重新上传</el-button>
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
              <p class="img-title">透视校正图</p>
              
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
                  <a :href="fixedUrl" download>下载透视校正图</a>
                </el-button>
                <el-button type="primary" v-if="showDownload" class="img-btn" @click="dialogVisibleT = true">查看溯源结果</el-button>
              </div>
            </div>
          </div>

          <div class="outer-container">
            <div id="total-control-panel">
              <el-form class="form-all" ref="form" :model="form" label-width="85px" :rules="rules">
                <el-form-item label="图片类型" prop="type">
                  <el-radio-group v-model="form.type" size="medium">
                    <el-radio border :label="1">截图</el-radio>
                    <el-radio border :label="2">照片</el-radio>
                  </el-radio-group>
                </el-form-item>
              </el-form>
              <el-button type="primary" style="margin-right: 20px;" :disabled="notAllowMark" @click="markUpload(2)">辅助标定后上传</el-button>
              <el-button type="primary" style="margin-right: 30px;" @click="showMarkDialog" :disabled="notAllowMark">手动标定</el-button>
              <el-button type="primary" style="margin-right: 40px; margin-top: 22px;" @click="markUpload(1)" :disabled="notAllowMark">直接上传泄漏图并查看解码溯源结果</el-button>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>

  <div id="dialog-container">
    <el-dialog
      class="dialog-class"
      top="7vh"
      width="50%"
      title="标定检测区域"
      :visible.sync="dialogVisible"
      :before-close="handleClose">
      <div id="marker-container">
        <img id="marked-img" :src="localUrl" alt="" style="margin: 0px; width: 100%">
      </div>
      <el-button type="primary" size="small" @click="delAllMarks" style="margin-top: 20px;">清空标点</el-button>

      <div slot="footer">
        <el-button @click="cancelMark" style="margin-right: 10px;">取消</el-button>
        <el-button type="primary" @click="markUpload(1)">确定上传</el-button>
      </div>
    </el-dialog>
  </div>

  <div id="dialog-table">
    <el-dialog
      class="dialog-class"
      title="查询结果"
      :visible.sync="dialogVisibleT"
      width="45%"
      top="7vh"
      :before-close="handleCloseT">
      <div>
        <el-table stripe :data="infoList" class="table" height=475>
          <el-table-column prop="id" label="ID" width="150" fixed></el-table-column>
          <el-table-column prop="timeStamp" label="时间" width="auto"></el-table-column>
          <el-table-column prop="ip" label="信息" width="auto"></el-table-column>
        </el-table>
      </div>
      <div slot="footer">
        <el-button type="primary" @click="dialogVisibleT = false">确定</el-button>
      </div>
    </el-dialog>
  </div>
</div>
</template>

<script>
export default {
  name: "DecoderPic",
  data() {
    return {
      isShowUpload: true,
      isShowImgUpload: false,
      
      localUrl: null,
      fixedUrl: null,

      form: {
        fileBase64: null,

        // 1: 截图
        // 2: 照片
        type: null,
      },

      dialogVisible: false,
      dialogVisibleT: false,

      // 变换坐标
      positionList: [],
      // 点的颜色
      pointColor: 'red',
      // 点的大小 
      pointSize: 10,

      // 查库结果显示
      infoList: [],

      rules: {
        type: [{ required: true, message: '请确认图片类型!', trigger: 'blur' }]
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
      this.form.fileBase64 = null;
      this.form.type = null;
      this.positionList = [];
      this.infoList = [];
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
    // 上传图片（autoVal: 1: 不自动标定; 2: 自动标定）
    markUpload(autoVal) {
      if (this.localUrl == null) {
        this.$message({
          message: "请先上传泄漏图!",
          type: 'warning',
          showClose: true
        });
        return;
      }
      if (this.form.type === null || this.form.type < 0 || this.form.type > 2) {
        this.$message({
          message: "请选择泄漏图类型!",
          type: 'warning',
          showClose: true
        });
        return;
      }
      if (this.positionList.length === 0 || this.positionList.length === 4) {
        const params = {
          fileBase64: this.form.fileBase64,
          // positions: this.positionList.length === 0 ? this.$res.defaultPos : this.positionList
          positions: this.positionList.length === 0 ? "" : this.positionList,
          type: this.form.type,
          auto: autoVal
        };
        this.$http.uploadPic(params).then(res => {
          const data = res.data;
          if (data.code === 200) {
            this.$message({
              message: data.msg,
              type: 'success',
              showClose: true
            });
            
            this.dialogVisible = false;
            this.delAllMarks();

            this.infoList = data.data;
            this.fixedUrl = this.$func.createDownloadFileUrl("decodedPic.jpg", data.fixedImg);

            this.dialogVisibleT = true;
          }
        }).catch(err => {
          this.$message({
            message: '上传失败!',
            type: 'error',
            showClose: true
          }); 
          console.log(err);
        });
      } else {
        this.$message({
          message: "请 不标定直接上传 或者 标定满 4 个点后再上传 !",
          type: 'warning',
          showClose: true
        });
      }
    },
    // 画点
    createMarker(x, y) {
      let div = document.createElement('div');
      div.className = 'marker';
      div.id = 'marker' + this.positionList.length;
      y = y + document.getElementById('marked-img').offsetTop - this.pointSize / 2;
      x = x + document.getElementById('marked-img').offsetLeft - this.pointSize / 2;
      div.style.width = this.pointSize + 'px';
      div.style.height = this.pointSize + 'px';
      div.style.backgroundColor = this.pointColor;
      div.style.left = x + 'px';
      div.style.top = y + 'px';

      this.delMarker(div);

      document.getElementById('marker-container').appendChild(div);
    },
    // 删除单个点（阻止冒泡行为和默认右键菜单事件）
    delMarker(div) {
      div.oncontextmenu = ((e) => {
        let id = e.target.id;
        document.getElementById('marker-container').removeChild(div);
        // 6 为 'marker' 字符数
        this.positionList = this.positionList.filter(item => item.id != id.slice(6, id.length));
        if (e && e.preventDefault) {
          // 阻止默认浏览器动作(W3C)
          e.preventDefault();
        } else {
          // IE中阻止函数器默认动作的方式
          window.event.returnValue = false;
        }
        return false;
      });
    },
    // 删所有点
    delAllMarks() {
      let myNode = document.getElementsByClassName("marker");
      let count = this.positionList.length;

      while (count > 0) {
        document.getElementById('marker-container').removeChild(myNode[count - 1]);
        count--;
      }
      this.positionList = [];
    },
    // 初始化标定框
    initMarker() {
      // 阻止冒泡行为和默认右键菜单事件
      document.getElementById('marked-img').oncontextmenu = ((e) => {
        if (e && e.preventDefault) {
          // 阻止默认浏览器动作(W3C)
          e.preventDefault();
        } else {
          // IE中阻止函数器默认动作的方式
          window.event.returnValue = false;
        }
        return false;
      });

      document.getElementById('marked-img').onmousedown = ((e) => {
        if (this.positionList.length === 4) {
          this.$message({
            message: "最多只能标定 4 个点!",
            type: 'warning',
            showClose: true
          });
          return;
        }

        e = e || window.event;
        if (e.button !== 2) {       //判断是否右击
          if (this.dialogVisible) {    //判断是否可以进行标注
            let x = e.offsetX || e.layerX;
            let y = e.offsetY || e.layerY;

            // 图片在 img 标签内的标签坐标 (x, y)
            // console.log(x, y);

            // 这里显示的是标签坐标相对于全图的比例坐标
            let myImg = document.querySelector("#marked-img");
            let currWidth = myImg.clientWidth;
            let currHeight = myImg.clientHeight;
            let ProportionWidthInImg = x / currWidth;
            let ProportionHeightInImg = y / currHeight;
            // console.log("图片比例高度：" + ProportionHeightInImg);
            // console.log("图片比例宽度：" + ProportionWidthInImg);

            // 这里保存的是比例坐标，用来传到后端来做高精度计算（首先还是要进行数值校正）
            if (ProportionWidthInImg > 1.0) ProportionWidthInImg = 1.0;
            if (ProportionWidthInImg < 0.0) ProportionWidthInImg = 0.0;
            if (ProportionHeightInImg > 1.0) ProportionHeightInImg = 1.0;
            if (ProportionHeightInImg < 0.0) ProportionHeightInImg = 0.0;
            this.positionList.push({
              id: this.positionList.length + 1,
              x: ProportionWidthInImg,
              y: ProportionHeightInImg
            });

            // 这里保存的是 img 标签坐标
            this.createMarker(x, y);
          }
        }
      });
    },
    // 展示标定框
    showMarkDialog() {
      this.dialogVisible = true;
      // 等待 DOM 渲染完成后再调用 标定初始化方法，直接调用初始化方法会报错：找不到 DOM 元素
      this.$nextTick(() => {
          this.initMarker();
        });
    },
    // 标定框关闭前询问
    handleClose(done) {
      this.$confirm('确认关闭?')
        .then(_ => {
          this.delAllMarks();
          done();
        })
        .catch(_ => {});
    },
    // 取消标定
    cancelMark() {
      this.dialogVisible = false;
      this.delAllMarks();
    },
    // 结果框关闭前询问
    handleCloseT() {
      this.$confirm('确认关闭?')
        .then(_ => {
          this.dialogVisibleT = false;
          done();
        })
        .catch(_ => {});
    },
  },
  mounted() {
    // 检查是否从图片攻击部分携带图片跳转而来
    let storedPic = this.$store.getters.getPsPicToDecode;
    if (storedPic) {
      this.isShowUpload = false;
      this.isShowImgUpload = true;
      this.form.fileBase64 = storedPic;
      this.localUrl = this.$func.createDownloadFileUrl("tempPsPic.jpg", storedPic);
    }
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
    // 是否允许标定
    notAllowMark() {
      let isEmpty = null;
      if (this.form.type === 1 || this.form.type === 2) {
        isEmpty = false;
      } else {
        isEmpty = true;
      }
      return (isEmpty || this.isShowUpload);
    }
  },
  beforeDestroy() {
    // 销毁实例前清除图片
    this.$store.commit('rmPsPicToDecode');
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
    margin-top: 361.2px;
    margin-left: 30px;
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
.dialog-class {
  text-align: center;
}
#dialog-table {
  text-align: center;
}
</style>

<style lang="scss">
// 该 style 下表示标定的点的样式；不可在 style 标签内添加 scoped，否则不生效
#marker-container {
  position: relative;
  img {
    border: solid 1px #000;
    display: inline-block;
    margin: 100px 100px;
    z-index: 1;
  }
  .marker {
    position: absolute;
    border-radius: 50%;
    z-index: 999;
  }
}
</style>

<style scoped>
.dialog-class /deep/ .el-dialog__body {
  padding-top: 10px;
  padding-bottom: 10px;
}
</style>