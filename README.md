#  Mark-Probe 防泄漏暗水印

## 简介

第十三届中国大学生服务外包创新创业大赛——A06云桌面的暗水印方案参赛作品

4.14 初赛提交

5.16 排名 6/10（即，东部赛区共10支队伍提交作品，分配名额为1个一等、2一个二等、2个三等，遗憾落榜）

初赛评语：

1. 项目完成度高，嵌入水印到水印提取都有工程实现，水印性能、抗性分析详细。
2. 项目描述基本清晰，整体目标规划和工作进度安排较合理，技术路线较清晰。
3. 系统设计防泄漏暗水印概要介绍方案设计创新描述简单，创新性不足，应用条件和环境描述较弱，市场应用前景说明不足。

> 演出效果和文档撰写不达标，已停止维护。
>
> まだまだです

##  快速使用

1. 将项目拉到本地，并且安装依赖

   ```bash
   git clone https://github.com/NephrenCake/Mark-Probe.git
   cd Mark-Probe
   pip install -r requirements.txt
   ```

2. 将模型权重放在 `Mark-Probe/weight` 下

3. 运行编码程序

   ```bash
   python ./tools/encode.py --img_path test/test_source/COCO_train2014_000000000009.jpg --model_path weight/infer.pth --output_path out/ --user_id 114514
   ```

4. 运行解码程序

   ```bash
   python ./tools/decode.py --img_path out/encoded.jpg --model_path weight/infer.pth
   ```

5. 运行检测并解码

   ```bash
   python ./tools/detect.py
   ```

6. 启动前后端服务 [这里](service/doc/README.md)
