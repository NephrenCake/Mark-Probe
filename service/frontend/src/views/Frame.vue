<template>
  <div>
    <el-container class="frame-container">
      <el-header class="header">
        <div class="header-content">
          <router-link :to=tolink class="ref">{{word}}</router-link>
        </div>
      </el-header>
      <el-main class="main">
        <router-view></router-view>
      </el-main>
      <el-footer class="footer">
        <div class="info">
          <span>2022 服务外包 A06&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
          <a href="https://github.com/NephrenCake/HiddenWatermark/" id="github" target="_blank">点击访问本项目 Github</a>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: "Frame",
  data() {
    return {
    }
  },
  methods: {
  },
  computed: {
    word: function() {
      if (this.$route.path === "/encoder") {
        return "To Decoder";
      } else {
        return "To Encoder";
      }
    },
    tolink: function() {
      if (this.$route.path.indexOf("/encoder") === 0) {
        return "/decoder";
      } else {
        return "/encoder";
      }
    }
  }
};
</script>

<style lang="scss" scoped>
.frame-container {
  height: 100vh;

  .header {
    background: #e3e3e3;
    height: 60px !important;
    padding: 0px !important;

    .header-content {
      overflow: hidden;
      float: right;
      padding: 15px;
    }
  }
  .main {
    background: #e3e3e3;
  }
}
.footer{
  height: 40px !important;
  text-align: center;
  border-top: 1px solid #d8dce5;
  background: #eee;
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
.ref {
  position: relative;
  text-decoration: none;
  font-size: 20px;
  color: #888;
  font-family: 'HOS Medium';
}
.ref:before{
  content: "";
  position: absolute;
  left: 0;
  bottom: -2px;
  width: 100%;
  height: 1.5px;
  background: #999;
  transform: scale(0);
  transition: all .3s;
}
.ref:hover:before {
  transform: scale(1);
}
</style>

<style scoped>
body /deep/ .el-main {
  padding: 0px;
}
</style>