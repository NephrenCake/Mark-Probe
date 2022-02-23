<template>
  <div>
    <div id="container">
      <el-row :gutter="0">
        <el-col :span="14" :offset="5">
          <el-card id="card">
            <div id="welcome">
              <p>Mark-Probe&nbsp;防泄漏暗水印系统</p>
              <p>演示程序 (Web 端)</p>
              <p class="team">团队编号: 2101421</p>
              <p class="team" style="margin-top: 10px;">团队名称: <span>永远不要抛弃僚机</span></p>
              <p>演示日期: {{date}}</p>
            </div>
            <div id="version">
              <p>系统版本号: V 1.0</p>
              <p>演示程序版本号: V 1.0</p>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div> 
  </div>
</template>

<script>
export default {
  name: "Main",
  computed: {
    date: function () {
      return this.$func.getDate();
    }
  }
}
</script>

<style lang="scss" scoped>
#container {
  margin-top: 50px;
}
#card {
  text-align: center;
}
#welcome {
  margin: 30px;
  font-size: 50px;
  color: #000;

  p:nth-child(1) {
    font-family: 'HOS Medium';
  }
  p:nth-child(2) {
    margin-top: 20px;
    font-size: 35px;
  }
  .team {
    margin-top: 60px;
    font-size: 25px;
  }
  p:nth-child(5) {
    margin-top: 60px;
    font-size: 20px;
  }
}
#version {
  margin-right: 30px;
  font-size: 10px;
  text-align: right;
}
</style>