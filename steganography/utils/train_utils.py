# -- coding: utf-8 --
import time

import torch
import torchvision
import torch.nn.functional as nn_F
import torchvision.transforms.functional as transforms_F
import logging

from kornia.color import rgb_to_hsv, rgb_to_yuv
from torchvision.utils import make_grid
from steganography.utils.distortion import make_trans, make_trans_2


def process_forward(Encoder,
                    Decoder,
                    lpips,
                    data,
                    scales,
                    cfg):
    img = data["img"].to(cfg.device)
    msg = data["msg"].to(cfg.device)

    # ------------------forward
    # Encoder
    res_img = Encoder({"img": img, "msg": msg})
    encoded_img = img + res_img
    encoded_img = torch.clamp(encoded_img, 0., 1.)

    # transform
    # todo 一个分支实现整体的识别，一个分支实现局部的识别
    # transformed_img, no_stretched_img, startpoints = make_trans(encoded_img, scales)
    transformed_img = make_trans_2(encoded_img, scales)

    # Decoder
    msg_pred, stn_img = Decoder(transformed_img, use_stn=scales['stn_loss'] == 1)  # stega

    # ------------------loss
    # todo 可以尝试使用其他实现方式，高斯模糊并不是最佳选择，期望标定的高频区域范围更多一些
    weight_mask = torch.abs(img - transforms_F.gaussian_blur(img, [7, 7], [10, 10]))
    weight_mask = torch.max(weight_mask) - weight_mask  # low weight in high frequency

    img_loss = torch.zeros(1).to(cfg.device)
    if scales["rgb_loss"] != 0:
        rgb_loss = mse_loss(encoded_img, img, mask=weight_mask)
        img_loss = img_loss + rgb_loss * scales["rgb_loss"]
    if scales["hsv_loss"] != 0:
        hsv_loss = mse_loss(rgb_to_hsv(encoded_img), rgb_to_hsv(img), mask=weight_mask)
        img_loss = img_loss + hsv_loss * scales["hsv_loss"]
    if scales["yuv_loss"] != 0:
        yuv_loss = mse_loss(rgb_to_yuv(encoded_img), rgb_to_yuv(img), mask=weight_mask)
        img_loss = img_loss + yuv_loss * scales["yuv_loss"]
    if scales["lpips_loss"] != 0:
        lpips_loss = lpips(img, encoded_img).mean()
        img_loss = img_loss + lpips_loss * scales["lpips_loss"]

    msg_loss = nn_F.binary_cross_entropy(msg_pred, msg)  # size(B, 100)的msg，二分类0和1

    if torch.ge(msg_loss, 0.6) and torch.le(img_loss, 0.01):
        loss = msg_loss  # 前期如果decoder能力提不上来，则encoder等待
    else:
        loss = img_loss + msg_loss
    # test
    # print(msg_loss)
    # msg_loss = scales["msg_loss"] * msg_loss
    # msg_loss.backward()
    # print(scales["msg_loss"], StegaStampEncoder.residual.weight.grad[0][0][0][0])
    # input("wait")

    # ------------------compute acc
    bit_acc, str_acc = get_msg_acc(msg, msg_pred)

    vis_img = {"res_img": res_img.data, "encoded_img": encoded_img.data,
               "transformed_img": transformed_img.data, "no_stretched_img": encoded_img.data,
               "stn_img": stn_img, }
    # todo 需要加入一个带纠正的准确率计算
    metric_result = {"loss": loss,
                     "img_loss": img_loss, "msg_loss": msg_loss,
                     "bit_acc": bit_acc, "str_acc": str_acc, }
    return metric_result, vis_img


def make_null_metric_dict():
    METRIC_LIST = ["loss", "img_loss", "msg_loss", "bit_acc", "str_acc"]
    return 0, {i: 0 for i in METRIC_LIST}


def train_one_epoch(Encoder,
                    Decoder,
                    lpips,
                    optimizer,
                    scheduler,
                    data_loader,
                    epoch,
                    tb_writer,
                    cfg):
    Encoder.train()
    Decoder.train()

    count, results = make_null_metric_dict()
    optimizer.zero_grad()
    start = time.time()
    for cur_iter, data in enumerate(data_loader):
        scales = cfg.get_cur_scales(cur_iter=cur_iter, cur_epoch=epoch)
        # ------------------forward & loss
        metric_result, _ = process_forward(Encoder,
                                           Decoder,
                                           lpips,
                                           data,
                                           scales,
                                           cfg)

        # ------------------backward
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

    return


@torch.no_grad()
def evaluate_one_epoch(Encoder,
                       Decoder,
                       lpips,
                       data_loader,
                       epoch,
                       tb_writer,
                       cfg):
    Encoder.eval()
    Decoder.eval()

    count, result = make_null_metric_dict()
    vis_img = {}
    iters = len(data_loader)
    for cur_iter, data in enumerate(data_loader):
        scales = cfg.get_cur_scales(cur_iter=cfg.iter_per_epoch, cur_epoch=epoch)
        # ------------------forward & loss
        metric_result, vis_img = process_forward(Encoder,
                                                 Decoder,
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
