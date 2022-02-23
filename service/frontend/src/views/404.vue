<template>
  <div>
    <div id="container">
      <el-row :gutter="0">
        <el-col :span="10" :offset="7">
          <el-card id="card" style="padding-left: 30px; padding-bottom: 10px;">
            <div>
              <p id="title">404</p>
              <p id="content">你的页面走丢了!</p> 
              <router-link to="/" class="ref">点击此处返回首页</router-link>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NotFound'
}
</script>

<style lang="scss" scoped>
#container {
  background: #e3e3e3;

  #card{
    margin-top: 80px;
  }
}
#title {
  font-family: "HOS Medium";
  font-size: 120px;
}
#content {
  font-size: 50px;
  margin-bottom: 50px;
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
  background: #888;
  transform: scale(0);
  transition: all .3s;
}
.ref:hover:before {
  transform: scale(1);
}
</style>