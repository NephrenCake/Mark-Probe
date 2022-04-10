import torch
from torch import nn
import torch.nn.functional as F

from steganography.models.swin import SwinTransformer


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()


class MPEncoder(nn.Module):
    """
    Mark-Probe-Encoder
    todo Upsample -> Repeat
        [由于已经扩大了感受野，暂时先不改]
    dilation -> (2, 2) 一个空洞
        [我们发现StegaStamp不能很好地实现局部识别的一大原因在于模型感受野不足。
        使用更大的卷积核5*5+2dilation来提取最初的信息]
        - 已修改开头3*3卷积为空洞卷积
        - 已修改up阶段2*2卷积为空洞卷积
    todo 删去 conv5 up6 conv6
    todo bn
        [待定]
    """

    def __init__(self, msg_size=96, img_size=448):
        super(MPEncoder, self).__init__()
        self.msg_dense_size = int(img_size / 8)
        msg_dense_out = int(img_size * img_size * 3 / (8 * 8))

        self.msg_dense = nn.Sequential(nn.Linear(msg_size, msg_dense_out),
                                       nn.LeakyReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=(5, 5), padding=(6, 6), dilation=(3, 3)),
                                   nn.LeakyReLU(inplace=True))
        # down
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.LeakyReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                   nn.LeakyReLU(inplace=True))
        # up
        self.up6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(2, 2), padding=(1, 1), dilation=(2, 2)),
                                 nn.LeakyReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
                                   nn.LeakyReLU(inplace=True))  # 7
        self.up7 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(2, 2), padding=(1, 1), dilation=(2, 2)),
                                 nn.LeakyReLU(inplace=True))  # 8
        self.conv7 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
                                   nn.LeakyReLU(inplace=True))  # 9
        self.up8 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(2, 2), padding=(1, 1), dilation=(2, 2)),
                                 nn.LeakyReLU(inplace=True))  # 10
        self.conv8 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
                                   nn.LeakyReLU(inplace=True))  # 11
        self.up9 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(2, 2), padding=(1, 1), dilation=(2, 2)),
                                 nn.LeakyReLU(inplace=True))  # 12
        self.conv9 = nn.Sequential(nn.Conv2d(70, 32, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2)),
                                   nn.LeakyReLU(inplace=True))  # 13
        # encode_output_file
        self.residual = nn.Conv2d(32, 3, kernel_size=(1, 1))  # 13

        self.up_x2 = nn.Upsample(scale_factor=2)
        self.up_x8 = nn.Upsample(scale_factor=8)

        initialize_weights(self)

    def forward(self, inputs):
        image, msg = inputs["img"], inputs["msg"]

        msg = self.msg_dense(msg)  # msg 96 => 9408
        msg = msg.view(image.size()[0], 3, self.msg_dense_size, self.msg_dense_size)  # msg 56*56*3

        inputs = torch.cat((self.up_x8(msg), image), dim=1)  # inputs 448*448*(3+3)
        conv1 = self.conv1(inputs)  # conv1 448*448*32

        conv2 = self.conv2(conv1)  # conv2 224*224*32
        conv3 = self.conv3(conv2)  # conv3 112*112*64
        conv4 = self.conv4(conv3)  # conv4 56*56*128
        conv5 = self.conv5(conv4)  # conv5 28*28*256

        up6 = self.up6(self.up_x2(conv5))  # up6 28*28*256->56*56*128
        merge6 = torch.cat((conv4, up6), dim=1)  # merge6 56*56*256
        conv6 = self.conv6(merge6)  # conv6 56*56*128
        up7 = self.up7(self.up_x2(conv6))  # up7 112*112*128->112*112*64
        merge7 = torch.cat((conv3, up7), dim=1)  # merge7 112*112*128
        conv7 = self.conv7(merge7)  # conv7 112*112*64
        up8 = self.up8(self.up_x2(conv7))  # up8 224*224*64->224*224*32
        merge8 = torch.cat((conv2, up8), dim=1)  # merge8 224*224*64
        conv8 = self.conv8(merge8)  # conv8 224*224*32
        up9 = self.up9(self.up_x2(conv8))  # up9 448*448*32->448*448*32
        merge9 = torch.cat((conv1, up9, inputs), dim=1)  # merge9 448*448*(32+32+6)
        conv9 = self.conv9(merge9)  # conv9 448*448*32

        return self.residual(conv9)  # residual 448*448*3


class MPDecoder(nn.Module):
    """
    Mark-Probe-Decoder
    已修改，加入了 "swin" or "conv" 可选 decoder
    """

    def __init__(self, msg_size=96, img_size=448, decoder_type="swin", has_stn=True):
        super(MPDecoder, self).__init__()
        self.has_stn = has_stn
        if has_stn:
            self.stn = STN(img_size=img_size)

        if decoder_type == "conv":
            self.decoder = ConvDecoder(msg_size=msg_size, img_size=img_size)
        elif decoder_type == "swin":
            self.decoder = SwinDecoder(msg_size=msg_size)

    def forward(self, x, use_stn=True, return_stn_img=True):
        # 需要待 decoder 部分稳定才可以开启 STN
        if use_stn and self.has_stn:
            x = self.stn(x)

        if not return_stn_img:
            return self.decoder(x)
        return self.decoder(x), x


class SwinDecoder(nn.Module):
    """
    已修改，可以自适应不同尺寸图
    """

    def __init__(self, msg_size=96, **kwargs):
        super(SwinDecoder, self).__init__()

        self.decoder = nn.Sequential(
            SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=msg_size,
                            **kwargs),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class ConvDecoder(nn.Module):
    """
    已修改，可以自适应不同尺寸图
    """

    def __init__(self, msg_size=96, img_size=448):
        super(ConvDecoder, self).__init__()
        img_conv_size = int((img_size - 1) / 32) + 1
        self.linear_num = img_conv_size * img_conv_size * 128

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.linear_num, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, msg_size),
            nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, x):
        return self.decoder(x)


class STN(nn.Module):
    """
    已修改原先的三层卷积为四层
        [为了减少参数 41054150(400*400) -> 51474374(448*448) -> 26079430(448*448) ]
    已修改，可以自适应不同尺寸图
    """

    def __init__(self, img_size=448):
        super(STN, self).__init__()
        down_sample = 32
        final_linear = 128
        self.fc_loc_num = int((img_size / down_sample) ** 2 * final_linear)

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_num, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 2)
        )

        initialize_weights(self)
        # 使用身份转换初始化权重/偏差
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x):
        # 空间变换器网络转发功能
        xs = self.localization(x).reshape(-1, self.fc_loc_num)
        theta = self.fc_loc(xs).reshape(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x
