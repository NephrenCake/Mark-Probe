# model

这里包括 MPNet（也就是自己的实现 MarkProbeNet）、StegaStamps 的原版实现、SwinTransformer 模块

## MPNet

可用模型包括：MPEncoder、MPDecoder。

在这里不存在 Discriminator，Discriminator 可以弃用。

默认参数将为 **msg_size=96 img_size=448**。

- [ ] Encoder：Upsample -> Repeat [由于已经扩大了感受野，暂时先不改]
- [x] Encoder：将更多的卷积核改为空洞卷积
  - 修改开头 3×3 卷积为 5×5 且空隙为 2 的空洞卷积
  - 修改 up 阶段 2×2 卷积为空洞卷积
  - 可以说：我们发现 StegaStamp 不能很好地实现局部识别的一大原因在于模型感受野不足。并且使用更大的卷积核5*5+2dilation来提取最初的信息，这一点的有效性在 ConvNeXt 有提及
- [ ] Encoder：删去 conv5 up6 conv6 [删去会导致感受野下降，搁置]
- [ ] Encoder：BN 层 [由于已经扩大了感受野，暂时先不改]
- [x] Decoder：加入了 "swin" or "conv" 可选 decoder
  - 现在可以使用 SwinTransformer
  - 在前馈过程中，依然可以使用 use_stn 来控制是否启用 STN 
- [ ] Decoder：ConvDecoder、STN 部分将作为 MPDecoder 的子模块，在 MPDecoder 中进行工作。
  - 它们现在都支持在初始化阶段自适应不同尺寸图
  - 注意，一旦初始化阶段完毕，后续的推理过程的尺寸大小仍然都应该按照初始化设定来
- [x] STN：增加卷积层，减少全连接数
  - 需要实验验证有效性

调用：

```python
import torch
import torchvision
import torch.nn.functional as F
from steganography.models.MPNet import MPEncoder, STN, MPDecoder

msg_size = 96
img_size = 448


def test_encoder_model():
  img = torch.randn((8, 3, 448, 448)).to("cuda")
  msg = torch.randn(8, 96).to("cuda")
  net = MPEncoder().to("cuda")
  output = net({"img": img, "msg": msg})
  print(output.size())
  print(sum(p.numel() for p in net.parameters()))  # stega 1739999  MPEncoder 1713071


def test_stn():
  img = torch.randn((8, 3, 448, 448))
  net = STN()
  output = net(img)
  print(output.size())
  print(sum(p.numel() for p in net.parameters()))  # 41054150 -> 26079430


def test_decoder_model():
  img = torch.randn((8, 3, img_size, img_size)).to("cuda")
  msg = torch.randn((8, msg_size)).to("cuda")
  net = MPDecoder(msg_size=msg_size, img_size=img_size, decoder_type="swin").to("cuda")
  output = net(img)
  print(output[0].size(), msg.size())
  msg_loss = F.binary_cross_entropy(output[0], msg)
  print(msg_loss)
  msg_loss.backward()
  print(sum(p.numel() for p in net.parameters()))  # stega 52505482  conv 39298182 swin 53672608
```

模型大小：

|                   | MPNet            | StegaStamps      |
| ----------------- |------------------| ---------------- |
| Encoder           | 1713071          | 1739999          |
| Decoder           | 53672608         | 52505482         |
| STN               | 26079430         | 41054150         |
| Decoder（无 STN） | 27593178（Swin-T） | 13218752（Conv） |

ps：由于尺寸从 400×400 改为 448×448.数据可能有些出入

## stega_net

emm，就是 StegaStamp 的原汁原味 pytorch 实现。不过大概率后面是不会用的。

> https://github.com/tancik/StegaStamp

包括 StegaStampEncoder、StegaStampDecoder、Discriminator。

- 还有一个带透视变换参数预测的 Decoder 版本，后期我会移除。
- Discriminator 后期也不会再用。

调用：

```python
from steganography.models import stega_net

Encoder = stega_net.StegaStampEncoder().to("cuda")  # stega
Decoder = stega_net.StegaStampDecoder().to("cuda")  # stega
Discriminator = stega_net.StegaStampDiscriminator().to("cuda")

img = torch.randn((8, 3, 400, 400)).to("cuda")
msg = torch.randn(8, 100).to("cuda")

res_img = Encoder({"img": img, "msg": msg})
msg_pred, stn_img = Decoder(img)
dis_pred = Discriminator(img)
```

注意，StegaStamp 仅支持 400×400 尺寸图像、100 字符串。

