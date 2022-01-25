<template>
  <div>
    <div id="container">
      <el-row :gutter="0">
        <el-col :span="8" :offset="8">
          <el-card id="card">
            <div style="text-align: center; margin: 19px;">
              <p style="margin-bottom: 40px;">
                <a href="/encoder" class="ref">Encoder 演示</a>
              </p>
              <p>
                <a href="/decoder" class="ref">Decoder 演示</a>
              </p>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div> 

    <div class="footer">
      <div class="info">
        <span>2022 服务外包 A06&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
        <a href="https://github.com/NephrenCake/HiddenWatermark/" id="github" target="_blank">点击访问本项目 Github</a>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Main"
}
</script>

<style lang="scss" scoped>
.ref {
  position: relative;
  text-decoration: none;
  font-size: 30px;
  color: #000;
  font-family: 'HOS Medium';
}
.ref:before{
  content: "";
  position: absolute;
  left: 0;
  bottom: -2px;
  width: 100%;
  height: 1.5px;
  background: #000;
  transform: scale(0);
  transition: all .3s;
}
.ref:hover:before {
  transform: scale(1);
}
.footer{
  height: 39px !important;
  text-align: center;
  border-top: 1px solid #d8dce5;
  background: #eee;
  padding-left: 20px;
  padding-right: 20px;
  .info {
    margin: 12px 0;
    font-size: 14px;
    color: #888;
    #github {
      color: #888;
    }
    #github:hover {
      color: #aaa;
    }
  }
}
#container {
  height: calc(100vh - 40px);
  background: #e3e3e3;

  #card{
    margin-top: 30vh;
  }
}
</style>