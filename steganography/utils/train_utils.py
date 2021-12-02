# -- coding: utf-8 --
import time

import torch
import torchvision
import torch.nn.functional as nn_F
import torchvision.transforms.functional as transforms_F
import logging

from kornia.color import rgb_to_hsv, rgb_to_yuv
from torchvision.utils import make_grid
from steganography.utils.distortion import make_trans


# from pytorch_wavelets import DTCWTForward
# xfm = DTCWTForward(J=2, biort='near_sym_b', qshift='qshift_b').cuda()


def process_forward(Encoder,
                    Decoder,
                    Discriminator,
                    lpips,
                    data,
                    scales,
                    cfg):
    img = data["img"].to(cfg.device)
    msg = data["msg"].to(cfg.device)
    b, c, w, h = img.shape

    # ------------------forward
    # Encoder
    res_img = Encoder({"img": img, "msg": msg})
    encoded_img = img + res_img
    encoded_img = torch.clamp(encoded_img, 0., 1.)

    # transform
    transformed_img, no_stretched_img, startpoints = make_trans(encoded_img, scales)

    # Decoder
    msg_pred, stn_img = Decoder(transformed_img, use_stn=scales['stn_loss'] == 1)  # stega
    # msg_pred, stn_img, startpoints_predict = Decoder(transformed_img)  # per
    # msg_pred, stn_img, startpoints_predict = Decoder(transformed_img).sigmoid(), no_stretched_img, startpoints  # swin

    # Discriminator
    dis_label = torch.cat((torch.zeros(b), torch.ones(b)), dim=0).to(cfg.device)  # raw-0 encoded-1
    dis_pred = Discriminator(torch.cat((img, encoded_img), dim=0))

    # ------------------loss
    weight_mask = torch.abs(img - transforms_F.gaussian_blur(img, [7, 7], [10, 10]))
    weight_mask = torch.max(weight_mask) - weight_mask  # low weight in high frequency

    img_loss = torch.zeros(1).to(cfg.device)
    if scales["rgb_loss"] != 0:
        rgb_loss = mse_loss(encoded_img, img, mask=weight_mask)
        img_loss = img_loss + rgb_loss
    if scales["hsv_loss"] != 0:
        hsv_loss = mse_loss(rgb_to_hsv(encoded_img), rgb_to_hsv(img), mask=weight_mask)
        img_loss = img_loss + hsv_loss
    if scales["yuv_loss"] != 0:
        yuv_loss = mse_loss(rgb_to_yuv(encoded_img), rgb_to_yuv(img), mask=weight_mask)
        img_loss = img_loss + yuv_loss
    # if scales["dtcwt_loss"] != 0:
    #     dtcwt_loss = nn_F.mse_loss(xfm(encoded_img)[0], xfm(img)[0])
    #     img_loss = img_loss + dtcwt_loss
    if scales["lpips_loss"] != 0:
        lpips_loss = lpips(img, encoded_img).mean()
        img_loss = img_loss + lpips_loss

    # position_loss = nn_F.l1_loss(startpoints_predict, startpoints)
    msg_loss = nn_F.binary_cross_entropy(msg_pred, msg)  # size(B, 100)的msg，二分类0和1
    dis_loss = nn_F.binary_cross_entropy(dis_pred, dis_label)  # 包含原图和编码图，cat成size(2B, )预测是否为编码图

    loss = img_loss + msg_loss
    if torch.ge(dis_loss, 0.001):  # 当鉴别器在一定损失之内时，不对其进行优化
        loss = loss + dis_loss

    # test
    # print(msg_loss)
    # msg_loss = scales["msg_loss"] * msg_loss
    # msg_loss.backward()
    # print(scales["msg_loss"], StegaStampEncoder.residual.weight.grad[0][0][0][0])
    # input("wait")

    # ------------------compute acc
    bit_acc, str_acc = get_msg_acc(msg, msg_pred)

    vis_img = {"res_img": res_img.data, "encoded_img": encoded_img.data,
               "transformed_img": transformed_img.data, "no_stretched_img": no_stretched_img,
               "stn_img": stn_img, }
    metric_result = {"loss": loss,
                     # "position_loss": position_loss,
                     "img_loss": img_loss, "msg_loss": msg_loss, "dis_loss": dis_loss,
                     "bit_acc": bit_acc, "str_acc": str_acc, }
    return metric_result, vis_img


def make_null_metric_dict():
    METRIC_LIST = ["loss",
                   # "position_loss",
                   "img_loss", "msg_loss", "dis_loss", "bit_acc", "str_acc"]
    return 0, {i: 0 for i in METRIC_LIST}


def train_one_epoch(Encoder,
                    Decoder,
                    Discriminator,
                    lpips,
                    optimizer,
                    scheduler,
                    data_loader,
                    epoch,
                    tb_writer,
                    cfg):
    Encoder.train()
    Decoder.train()
    Discriminator.train()

    count, results = make_null_metric_dict()
    optimizer.zero_grad()
    start = time.time()
    for cur_iter, data in enumerate(data_loader):
        scales = cfg.get_cur_scales(cur_iter=cur_iter, cur_epoch=epoch)
        # ------------------forward & loss
        metric_result, _ = process_forward(Encoder,
                                           Decoder,
                                           Discriminator,
                                           lpips,
                                           data,
                                           scales,
                                           cfg)

        # ------------------backward
        # metric_result["position_loss"].backward(retain_graph=True)
        metric_result["loss"].backward()
        optimizer.step()

        # ------------------update
        for item in results:
            results[item] += metric_result[item].item()
        count += 1

        # ------------------打印当前iter的loss
        if cur_iter % cfg.log_interval == 0 and cur_iter != 0 or cur_iter == cfg.iter_per_epoch - 1:
            for item in results:
                results[item] /= count
            cur_cost_time, pre_cost_time = compute_time(start, cur_iter, cfg.iter_per_epoch)

            logging.info(f"epoch:[{epoch}/{cfg.max_epoch - 1}] iter:[{cur_iter}/{cfg.iter_per_epoch - 1}] "
                         f"train:{cur_cost_time}<{pre_cost_time} - "
                         # f"lr:{round(optimizer.param_groups[1]['lr'], 4)} "
                         f"loss:{round(results['loss'], 4)} "
                         f"img_loss:{round(results['img_loss'], 4)} "
                         f"msg_loss:{round(results['msg_loss'], 4)} "
                         # f"position_loss:{round(results['position_loss'], 4)} "
                         f"bit_acc:{round(results['bit_acc'], 4)} "
                         f"str_acc:{round(results['str_acc'], 2)} ")

            # 随时观察
            torchvision.utils.save_image(_["encoded_img"], 'encoded_img.jpg')
            torchvision.utils.save_image(_["stn_img"], 'stn_img.jpg')

            # tensorboard
            for k, v in results.items():
                tb_writer.add_scalars(f"data/{k}", {"train": v}, global_step=epoch * cfg.iter_per_epoch + cur_iter)
            for k, v in scales.items():
                tb_writer.add_scalar(f"scale/{k}", v, global_step=epoch * cfg.iter_per_epoch + cur_iter)
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"],
                                 global_step=epoch * cfg.iter_per_epoch + cur_iter)

            count, results = make_null_metric_dict()

        # ------------------更新lr
        optimizer.zero_grad()
        scheduler.step()
        # ------------------限制dis模型的参数大小，否则loss会反向升高
        for i in Discriminator.parameters():
            i.data = torch.clamp(i, -0.25, 0.25)

    return


@torch.no_grad()
def evaluate_one_epoch(Encoder,
                       Decoder,
                       Discriminator,
                       lpips,
                       data_loader,
                       epoch,
                       tb_writer,
                       cfg):
    Encoder.eval()
    Decoder.eval()
    Discriminator.eval()

    count, result = make_null_metric_dict()
    vis_img = {}
    iters = len(data_loader)
    for cur_iter, data in enumerate(data_loader):
        scales = cfg.get_cur_scales(cur_iter=cfg.iter_per_epoch, cur_epoch=epoch)
        # ------------------forward & loss
        metric_result, vis_img = process_forward(Encoder,
                                                 Decoder,
                                                 Discriminator,
                                                 lpips,
                                                 data,
                                                 scales,
                                                 cfg)

        # ------------------update
        for item in result:
            result[item] += metric_result[item].item()

    for item in result:
        result[item] /= iters

    # tensorboard
    for image_tag in vis_img.keys():
        image = make_grid(vis_img[image_tag], normalize=True, scale_each=True, nrow=4)
        tb_writer.add_image(image_tag, image, global_step=(epoch + 1) * cfg.iter_per_epoch)
    return result


def compute_time(start, cur_iter, iterations):
    cur_time_cost = time.time() - start
    pre_cost_time = cur_time_cost / (cur_iter + 1) * iterations - cur_time_cost
    cur_time_cost = time.strftime('%H:%M:%S', time.gmtime(cur_time_cost))
    pre_cost_time = time.strftime('%H:%M:%S', time.gmtime(pre_cost_time))
    return cur_time_cost, pre_cost_time


def get_msg_acc(msg_true, msg_pred):
    msg_pred = torch.round(msg_pred)
    # batch中二进制级预测正确的列表
    correct_pred = (msg_pred.size()[1]) - torch.count_nonzero(msg_pred - msg_true, dim=1)
    str_acc = 1.0 - torch.count_nonzero(correct_pred - (msg_pred.size()[1])) / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred) / (msg_pred.size()[0] * msg_pred.size()[1])
    return bit_acc, str_acc


def mse_loss(pre, tar, mask):
    return torch.mean((pre - tar) ** 2 * mask)
