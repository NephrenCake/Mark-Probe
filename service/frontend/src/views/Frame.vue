<template>
  <div>
    <el-container class="frame-container">
      <el-header class="header">
        <div class="header-content">
          {{word}}
        </div>
      </el-header>

      <el-main class="main">
        <div class="main-content">
          <router-view></router-view>
        </div>
      </el-main>

      <el-footer class="footer">
        <div class="info">
          <span>2022 服务外包 A06 2101421&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
          <span><span style="font-family: 'Arial';">©</span> 2022 Mark-Probe</span>
          <!-- <a href="https://github.com/NephrenCake/HiddenWatermark/" id="github" target="_blank">点击访问本项目 Github</a> -->
        </div>
      </el-footer>
    </el-container>

    <div :class="{navigation: true, active: nav_isOpen}">
      <div class="btn" @click="showMenu">
        <div id="nav_check"></div>
        <span></span>
        <span></span>
      </div>

      <div class="menu">
        <div class="menu_item">
          <router-link class="ref item_ref" to='/'>首页</router-link>
        </div>

        <div :class="{menu_item: true, has_sub: true, active: subNav_isOpen}" @click="showSubMenu($event)">
          <span class="ref">编码隐写演示</span>

          <div class="menu_item__i" style="margin-top: 5px;">
            <router-link class="ref item_ref" to='/encoder/pic'>图片编码演示</router-link>
          </div>
          <div class="menu_item__i">
            <router-link class="ref item_ref" to='/encoder/stream'>视频流编码演示</router-link>
          </div>
        </div>

        <!-- <div class="menu_item has_sub" @click="showSubMenu($event)"> -->
        <div class="menu_item">
          <!-- <span class="ref">解码溯源演示</span> -->
          <router-link class="ref item_ref" to='/decoder/pic'>解码溯源演示</router-link>

          <!-- <div class="menu_item__i" style="margin-top: 5px;">
            <router-link class="ref item_ref" to='/decoder/pic'>图片解码与溯源演示</router-link>
          </div>
          <div class="menu_item__i">
            <router-link class="ref item_ref" to='/decoder/stream'>视频流解码与溯源演示</router-link>
          </div> -->
        </div>

        <div class="menu_item">
          <router-link class="ref item_ref" to='/ps'>图像攻击演示</router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Frame",
  data() {
    return {
      nav_isOpen: false,
      subNav_isOpen: false
    }
  },
  methods: {
    // 关闭菜单栏
    closeNav() {
      // let activeNavList = document.querySelectorAll('.active');

      // // NodeList 转为 List
      // activeNavList = [].slice.call(activeNavList);
      // activeNavList.forEach(function(element) {
      //   element.classList.remove('active');
      // });
      this.subNav_isOpen = false;
      this.nav_isOpen = false;
    },
    // 菜单栏展示
    showMenu() {
      if (!this.nav_isOpen) {
        this.nav_isOpen = true;
      } else {
        this.subNav_isOpen = false;
        this.nav_isOpen = false;
      }
    },
    // 子菜单栏展示（单步调试后发现：该事件在每次 DOM 渲染完成后，会自发地调用一次，原因未知）
    showSubMenu(event) {
      if (!this.nav_isOpen) {
        this.subNav_isOpen = false;
      } else {
        this.subNav_isOpen = !this.subNav_isOpen;
      }
    },

    // 动态调整高度
    changeHeight() {
      let mainContainerHeight = document.getElementsByClassName('el-main main')[0].clientHeight;
      let mainContentHeight = document.getElementsByClassName('main-content')[0].clientHeight;
      document.getElementsByClassName('el-main main')[0].style.paddingTop = ((mainContainerHeight - mainContentHeight) / 2) + "px";
    }
  },
  computed: {
    word: function() {
      if (this.$route.path === "/encoder/pic") {
        return "图片编码演示";
      } else if (this.$route.path === "/encoder/stream") {
        return "视频流编码演示";
      } else if (this.$route.path === "/decoder/pic") {
        return "图片解码与溯源演示";
      } else if (this.$route.path === "/decoder/stream") {
        return "视频流解码与溯源演示";
      } else if (this.$route.path === "/ps") {
        return "图像攻击演示";
      }
    }
  },
  mounted() {
    let that = this;
    that.changeHeight();
    // window.onresize = this.$func.debounce(function() { 
    //   that.changeHeight();
    // }, 1000, true);

    window.onresize = () => {
      return (() => {
        that.changeHeight();
      })();
    };
  },
  watch: {
    $route: {
      handler: function(to, from) {
        // 在 DOM 渲染完成后执行导航初始化
        this.$nextTick(() => {
          this.closeNav();
          this.changeHeight();
        });
      },
      // 深度观察监听
      // deep: true
    }
  },
  destroyed() {
    window.onresize = null;
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
      color: #888;
      font-family: 'HOS Medium';
      font-size: 25px;
      position: absolute;
      overflow: hidden;
      float: left;
      left: 80px;
      padding-top: 20px;
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

.menu_item {
  width: 250px;
  height: 24px;
  padding: 10px 15px;
  font-family: "HOS";
  cursor: pointer;

  span {
    font-size: 20px;
  }

  .menu_item__i {
    display: none;
    height: 19px;
    width: 210px;
    padding: 5px 0px 5px 30px;
    
    .item_ref {
      font-size: 15px;
    }
  }
}
</style>

<style scoped>
body /deep/ .el-main {
  padding: 0px;
}

::-webkit-scrollbar {
  width: 0px !important;
}

.navigation {
	position: absolute;
	top: 8px;
	left: 10px;
	justify-content: center;
	align-items: center;
	background: #e3e3e3;
	padding: 0px;
	transition: 0.5s;
	border-radius: 4px;
	overflow: hidden;
}
.navigation:hover {
  box-shadow: 0 0 10px rgba(0,0,0,.1);
  background: #fff;
}
.navigation .menu {
  display: none;
  margin-bottom: 2px;
}
.navigation.active {
  box-shadow: 0 0 10px rgba(0,0,0,.1);
  background: #fff;
}
.navigation.active .menu {
  display: block;
}

.navigation.active .menu .has_sub::before {
  content: '';
  position: absolute;
  width: 6px;
  height: 6px;
  z-index: 100;
  right: 15px;
  margin-top: 5px;
  border: 2px solid #000;
  border-top: 2px solid #fff;
  border-right: 2px solid #fff;
  transform: rotate(-135deg);
  transition: 0.5s;
  pointer-events: none;
}
.navigation.active .menu .has_sub.active {
  animation-delay: 1s;
  -webkit-animation: has_sub_height 0.5s forwards;
  animation: has_sub_height 0.5s forwards;
}
.navigation.active .menu .has_sub.active::before {
  margin-top: 2px;
  transform: rotate(-45deg);
}
.navigation.active .menu .has_sub.active .menu_item__i {
  display: block;
  opacity: 0%;
  -webkit-animation: has_sub_item 0.5s forwards;
  animation: has_sub_item 0.5s forwards;
}

.navigation .btn {
  width: 60px;
}
.navigation .btn #nav_check {
  display: block;
	width: inherit;
	height: 50px;
	cursor: pointer;
	opacity: 0;
}
.navigation .btn span {
	position: absolute;
  top: 23px;
	left: 15px;
	width: 30px;
	height: 4px;
	background: #666;
	pointer-events: none;
	transition: 0.5s;
}
.navigation.active .btn span {
	background: #666;
}
.navigation .btn span:nth-child(2) {
	transform: translateY(-8px);
}
.navigation.active span:nth-child(2) {
	transform: translateY(0) rotate(45deg);
}
.navigation .btn span:nth-child(3) {
	transform: translateY(8px);
}
.navigation.active .btn span:nth-child(3) {
	transform: translateY(0) rotate(-45deg);
}

@keyframes has_sub_height {
  0% {height: 24px;}
  100% {height: 78px;}
}
@keyframes has_sub_item {
  0% {opacity: 0%;}
  30% {opacity: 1%;}
  100% {opacity: 100%;}
}
</style>