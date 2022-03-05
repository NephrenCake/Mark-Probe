#  Mark-Probe 防泄漏暗水印系统

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
   python ./tools/encode.py --img_path test/test_source/COCO_train2014_000000000009.jpg --model_path weight/latest-0.pth --output_path out/ --user_id 114514
   ```

4. 运行解码程序

   ```bash
   python ./tools/decode.py --img_path out/encoded.jpg --model_path weight/latest-0.pth
   ```

   

