<template>
  <div>
    <div id="container">
      <el-row :gutter="0">
        <el-col :span="10" :offset="7">
          <el-card id="card" style="padding-left: 30px; padding-bottom: 10px;">
            <div>
              <p id="title">404</p>
              <p id="content">你的页面走丢了!</p>
              <el-link type="info" @click="goMain" id="go-main">点击此处返回主页</el-link>
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
  methods: {
    goMain() {
      this.$router.push('/');
    }
  }
}
</script>

<style lang="scss" scoped>
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
    margin-top: 20vh;
  }
}
#go-main {
  font-size: 20px;
}
#title {
  font-family: "HOS Medium";
  font-size: 120px;
}
#content {
  font-size: 50px;
  margin-bottom: 50px;
}
</style>