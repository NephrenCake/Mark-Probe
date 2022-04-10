# -- coding: utf-8 --
import time
import warnings

warnings.filterwarnings("ignore")
import torch
import torchvision
import torch.nn.functional as nn_F
import torchvision.transforms.functional as transforms_F
import logging
import copy
from kornia.augmentation import RandomMixUp
from kornia.color import rgb_to_hsv, rgb_to_yuv
from torchvision.utils import make_grid
from steganography.utils.distortion import make_trans_for_photo, make_trans_for_crop, non_spatial_trans


# 超分没必要了。md
def process_forward(Encoder,
                    Decoder,
                    lpips,
                    data,
                    scales,
                    cfg):
    img = data["img"].to(cfg.device)  # 传进来的是需要超分图  origin image
    msg = data["msg"].to(cfg.device)
    mask = data["mask"].to(cfg.device)

    # img_low = transforms_F.resize(img, cfg.img_size)  # simulate the process of resize
    # ------------------forward
    # Encoder
    # limit = 0.5 - scales['clamp_limit']
    # res = torch.clamp(Encoder({"img": img, "msg": msg}), -limit, limit)  # res_image (448,448)  #the encoder_module start forward
    res = Encoder({"img": img, "msg": msg})
    # res_clamp = torch.clamp(res, -limit, limit)
    # res_high = transforms_F.resize(res_low, img.shape[-2:])  # res_low  -> resize to original size
    encoded_img = torch.clamp(img + res, 0., 1.)
    # del res_high

    # transform
    # trans_img = transforms_F.resize(encoded_img, cfg.img_size)
    # ----------------------非空间变换
    trans_img = non_spatial_trans(encoded_img, scales)
    # 一个分支实现整体识别的变换，一个分支实现局部识别的变换   the logic of distortion must be fixed especially jpeg_trans
    photo_img = make_trans_for_photo(trans_img, scales)
    crop_img = make_trans_for_crop(trans_img, scales)
    # del trans_img

    # Decoder  for the BalanceDataParallel Decoder is safe
    photo_msg_pred, stn_img = Decoder(photo_img, use_stn=scales['stn_loss'] == 1)
    crop_msg_pred, _ = Decoder(crop_img, use_stn=False)

    # ------------------loss
    img_loss = torch.zeros(1).to(cfg.device)
    if scales["rgb_loss"] != 0:
        rgb_loss = mse_loss(encoded_img, img, mask=mask)
        img_loss += rgb_loss * scales["rgb_loss"]
    if scales["hsv_loss"] != 0:
        hsv_loss = mse_loss(rgb_to_hsv(encoded_img), rgb_to_hsv(img), mask=mask)
        img_loss += hsv_loss * scales["hsv_loss"]
    if scales["yuv_loss"] != 0:
        yuv_loss = mse_loss(rgb_to_yuv(encoded_img), rgb_to_yuv(img), mask=mask)
        img_loss += yuv_loss * scales["yuv_loss"]
    if scales["lpips_loss"] != 0:
        lpips_loss = lpips(img, encoded_img).mean()
        img_loss += lpips_loss * scales["lpips_loss"]

    photo_msg_loss = nn_F.binary_cross_entropy(photo_msg_pred, msg)
    crop_msg_loss = nn_F.binary_cross_entropy(crop_msg_pred, msg)
    msg_loss = (photo_msg_loss + crop_msg_loss) / 2

    if torch.ge(msg_loss, 0.1) and torch.le(img_loss, 0.02):  # 前期如果decoder能力跟不上encoder，则encoder等待
        loss = msg_loss
    else:
        loss = img_loss + msg_loss

    # ------------------compute acc
    photo_bit_acc, photo_str_acc, photo_right_str_acc = get_msg_acc(msg, photo_msg_pred)
    crop_bit_acc, crop_str_acc, crop_right_str_acc = get_msg_acc(msg, crop_msg_pred)
    bit_acc = (photo_bit_acc + crop_bit_acc) / 2
    str_acc = (photo_str_acc + crop_str_acc) / 2
    right_str_acc = (photo_right_str_acc + crop_right_str_acc) / 2


    vis_img = {"res_low": res.data, "encoded_img": encoded_img.data,
               "photo_img": photo_img.data, "crop_img": crop_img.data,
               "stn_img": stn_img.data, }
    metric_result = {
        "loss": loss, "img_loss": img_loss, "msg_loss": msg_loss,
        "bit_acc": bit_acc, "str_acc": str_acc, "right_str_acc": right_str_acc,
        "photo_msg_loss": photo_msg_loss, "crop_msg_loss": crop_msg_loss,
        "photo_bit_acc": photo_bit_acc, "photo_str_acc": photo_str_acc, "photo_right_str_acc": photo_right_str_acc,
        "crop_bit_acc": crop_bit_acc, "crop_str_acc": crop_str_acc, "crop_right_str_acc": crop_right_str_acc,
        "lpips_loss": torch.tensor(0) if scales["lpips_loss"] == 0 else lpips_loss,
        "rgb_loss": torch.tensor(0) if scales["rgb_loss"] == 0 else rgb_loss,
        "hsv_loss": torch.tensor(0) if scales["hsv_loss"] == 0 else hsv_loss,
        "yuv_loss": torch.tensor(0) if scales["yuv_loss"] == 0 else yuv_loss,
    }
    return metric_result, vis_img


METRIC_LIST = ["loss", "img_loss", "msg_loss", "bit_acc", "str_acc", "right_str_acc",
               "photo_msg_loss", "crop_msg_loss",
               "photo_bit_acc", "photo_str_acc", "photo_right_str_acc",
               "crop_bit_acc", "crop_str_acc", "crop_right_str_acc",
               "lpips_loss", "rgb_loss", "hsv_loss", "yuv_loss"]


def make_null_metric_dict():
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
            # 更新 loss
            cfg.loss_ascending(results['right_str_acc'], 0.75, 0.5, 0.5, 0.5)

            logging.info(f"epoch:[{epoch}/{cfg.max_epoch - 1}] iter:[{cur_iter}/{cfg.iter_per_epoch - 1}] "
                         f"train:{cur_cost_time}<{pre_cost_time} - "
                         # f"lr:{round(optimizer.param_groups[1]['lr'], 4)} "
                         f"loss:{round(results['loss'], 4)} "
                         f"img_loss:{round(results['img_loss'], 4)} "
                         f"msg_loss:{round(results['msg_loss'], 4)} "
                         f"bit_acc:{round(results['bit_acc'], 4)} "
                         f"str_acc:{round(results['str_acc'], 2)} "
                         f"right_str_acc:{round(results['right_str_acc'], 2)}")

            # 随时观察

            torchvision.utils.save_image(_["encoded_img"], 'encoded_img.jpg')
            torchvision.utils.save_image(_["stn_img"], 'stn_img.jpg')
            # torchvision.utils.save_image(_["res_low"], 'res_img.jpg')

            # tensorboard
            tb_writer.add_histogram("res_channel_0_histogram", _["res_low"][:, 0, ...],
                                    global_step=epoch * cfg.iter_per_epoch + cur_iter)
            tb_writer.add_histogram("res_channel_1_histogram", _["res_low"][:, 1, ...],
                                    global_step=epoch * cfg.iter_per_epoch + cur_iter)
            tb_writer.add_histogram("res_channel_2_histogram", _["res_low"][:, 2, ...],
                                    global_step=epoch * cfg.iter_per_epoch + cur_iter)
            # for name, param in Decoder.state_dict().items():
            #     if "stn.localization" in name:
            #         tb_writer.add_histogram(tag=name+'_grad', values=param, global_step=epoch * cfg.iter_per_epoch + cur_iter)
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

        # ------------------可能的手操stn卷积层权重，但不建议
        # for i in Decoder.stn.localization.parameters():
        #     i.data = torch.clamp(i, -0.25, 0.25)

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
        # 添加res的histogram 添加 encoded_img 分布
        # if image_tag in ["res_low"]:  # ,"photo_img",
        #     tb_writer.add_histogram(image_tag + "_histogram", image, global_step=(epoch + 1) * cfg.iter_per_epoch)

    return result


def compute_time(start, cur_iter, iterations):
    cur_time_cost = time.time() - start
    pre_cost_time = cur_time_cost / (cur_iter + 1) * iterations - cur_time_cost
    cur_time_cost = time.strftime('%H:%M:%S', time.gmtime(cur_time_cost))
    pre_cost_time = time.strftime('%H:%M:%S', time.gmtime(pre_cost_time))
    return cur_time_cost, pre_cost_time


def get_msg_acc(msg_true, msg_pred):
    """
    str_acc 当一个batch中的msg_pred 与 msg_true 完全相等的时候 这一个msg才会被判定为正确
    right_str_acc 应为msg存在校验位 也就是本身信息有着校验的能力 当msg_pred 的预测在msg校验能力之内就可以判定该msg_pred 就是正确的。
    """
    msg_pred = torch.round(msg_pred)
    correct_pred = (msg_pred.size()[1]) - torch.count_nonzero(msg_pred - msg_true, dim=1)  # 正确了多少个位

    # 优化： correct_pred_copy 本质上是 correct_pred  tensor的数据进行加减运算的时候 将所有的内容都进行运算
    #       correct_pred_copy 中存放的是 一个batch中 的一行数据里面正确的位数
    # 只要有少于五个bit错误就可以认为正确 91 位correct 定义容忍位
    correct_pred_copy = copy.deepcopy(correct_pred)
    correct_pred_copy[correct_pred > msg_pred.size()[1] - 5] = msg_pred.size()[1]

    str_acc = 1.0 - torch.count_nonzero(correct_pred - (msg_pred.size()[1])) / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred) / (msg_pred.size()[0] * msg_pred.size()[1])
    right_str_acc = 1.0 - torch.count_nonzero(correct_pred_copy - (msg_pred.size()[1])) / correct_pred.size()[0]
    return bit_acc, str_acc, right_str_acc


def mse_loss(pre, tar, mask):
    return torch.mean((pre - tar) ** 2 * mask)
