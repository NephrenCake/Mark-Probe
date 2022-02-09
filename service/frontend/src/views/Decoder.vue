<template>
  <div>
    <el-card id="card-container">
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
          <div style="margin-bottom: 30px;">
            <el-button type="primary" :loading="false" size="small" @click="uploadButtonClick" style="margin-right: 10px;">重新上传</el-button>
            <el-button type="primary" :loading="false" size="small" @click="showMarkDialog" style="margin-right: 10px;">标定检测区域</el-button>
            <el-button type="primary" :loading="false" size="small" @click="processButtonClick">确定上传</el-button>
          </div>
          <div>
            <el-image
              style="height: 400px;"
              :src="localUrl"
              :preview-src-list="[localUrl]"
              fit="scale-down">
            </el-image>
          </div>
        </div>
      </div>
    </el-card>

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
          <el-button type="primary" @click="processButtonClick">确定上传</el-button>
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
          <el-table stripe :data="infoList" class="table" height="100%">
            <el-table-column prop="id" label="ID" width="150" fixed></el-table-column>
            <el-table-column prop="timeStamp" label="时间" width="auto"></el-table-column>
            <el-table-column prop="ip" label="IP" width="auto"></el-table-column>
          </el-table>
        </div>
        <div slot="footer">
          <el-button type="primary" @click="showEditedImg">显示处理后图片</el-button>
          <el-button type="primary" @click="dialogVisibleT = false">确定</el-button>
        </div>
      </el-dialog>
    </div>

    <el-dialog
      class="dialog-class"
      style="text-align: center;"
      title="处理后图片"
      :visible.sync="dialogVisibleE"
      width="45%"
      top="7vh">
      <div>
        <el-image
          style="height: 400px;"
          :src="editedImg"
          :preview-src-list="[editedImg]"
          fit="scale-down">
        </el-image>
      </div>
      <div slot="footer">
        <el-button type="primary" @click="closeEditedImg">返回</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
export default {
  name: "Decoder",
  data() {
    return {
      // 展示上传框
      isShowUpload: true,
      // 展示预览框
      isShowImgUpload: false,
      // 本地文件 URL (Blob 格式)
      localUrl: null,

      dialogVisible: false,
      dialogVisibleT: false,
      dialogVisibleE: false,

      // 变换坐标
      positionList: [],
      // 点的颜色
      pointColor: 'red',
      // 点的大小 
      pointSize: 10,  

      // 图片提交表单
      form: {
        fileBase64: null
      },

      // 查库结果显示
      infoList: [],

      // 处理后图片
      editedImg: null
    }
  },
  methods: {
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
      this.localUrl = null;
      this.positionList = [];
      this.isShowImgUpload = false;
      this.isShowUpload = true;
      this.editedImg = null;
    },
    // 上传图片与比例坐标信息
    processButtonClick() {
      if (this.positionList.length === 0 || this.positionList.length === 4) {
        const params = {
          fileBase64: this.form.fileBase64,
          // positions: this.positionList.length === 0 ? this.$res.defaultPos : this.positionList
          positions: this.positionList.length === 0 ? "" : this.positionList
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
            this.editedImg = "data:image/jpeg;base64," + data.editedImg;

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
    // 显示处理后图片
    showEditedImg() {
      this.dialogVisibleE = true;
    },
    // 关闭处理后图片预览
    closeEditedImg() {
      this.dialogVisibleE = false;
    }
  }
}
</script>

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

<style lang="scss" scoped>
#card-container {
  width: calc(100% - 400px);
  margin-left: 200px;
  height: calc(100vh - 142px);

  #upload-unit { 
    text-align: center;
    margin-top: 15%;
  }

  #preview {
    text-align: center;
    margin-top: 3%;
  }
}

#dialog-container {
  text-align: center;
}

#dialog-table {
  text-align: center;
}
</style>

<style scoped>
.dialog-class /deep/ .el-dialog__body {
  padding-top: 10px;
  padding-bottom: 10px;
}
</style>