# -- coding: utf-8 --
import os
import sys

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import torch

from steganography.models.CINet import CIDecoder, CIEncoder
from steganography.utils.dataset import get_dataloader
from steganography.utils.log_utils import get_logger
from steganography.utils.train_utils import train_one_epoch, evaluate_one_epoch
from steganography.config.config import TrainConfig

from lpips import LPIPS


def main():
    cfg = TrainConfig()
    cfg.check_cfg()

    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir=os.path.join(sys.path[0], cfg.tensorboard_dir, cfg.exp_name))
    logger = get_logger(cfg.save_dir, cfg.exp_name)
    # 打印配置参数
    for name, value in vars(cfg).items():
        logger.info(f"%-25s{value}" % (name + ":"))
    logger.info(f'Using device: {cfg.device}')
    logger.info(f'Using dataloader workers: {cfg.num_workers}')

    # 实例化数据集
    train_loader, val_loader = get_dataloader(img_set_list=cfg.img_set_list,
                                              img_size=cfg.img_size,
                                              msg_size=cfg.msg_size,
                                              val_rate=cfg.val_rate,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.num_workers)
    cfg.set_iteration(len(train_loader))
    logger.info(f'{cfg.iter_per_epoch} iterations per epoch.')

    # 实例化模型
    Encoder = CIEncoder(msg_size=cfg.msg_size,
                        img_size=cfg.img_size[0]).to(cfg.device)
    Decoder = CIDecoder(msg_size=cfg.msg_size,
                        img_size=cfg.img_size[0],
                        decoder_type="conv").to(cfg.device)
    lpips_metric = LPIPS(net="alex").to(cfg.device)

    # 定义优化器和学习率策略
    optimizer = torch.optim.Adam(params=[
        {'params': Encoder.parameters()},
        {'params': Decoder.parameters()},
    ], lr=cfg.lr_base, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if cfg.use_warmup:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=cfg.get_warmup_cos_lambda())
    else:
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: cfg.lr_min)

    # 如果存在预训练权重则载入
    epoch = 0
    if os.path.exists(cfg.resume) or os.path.exists(cfg.pretrained):
        if os.path.exists(cfg.resume):
            checkpoint = torch.load(cfg.resume, map_location=cfg.device)
            logger.info(f"resume from {cfg.resume}")
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch = checkpoint['epoch']
        else:
            checkpoint = torch.load(cfg.pretrained, map_location=cfg.device)
            logger.info(f"use pretrain-weights {cfg.pretrained}")

        for model in cfg.load_models:
            net_state_dict = eval(f"{model}.state_dict()")
            for k, v in net_state_dict.items():
                if k in checkpoint[model]:
                    net_state_dict[k] = checkpoint[model][k]
                else:
                    logger.warning(f"{model}: {k} not found")
            eval(f"{model}.load_state_dict(net_state_dict)")
    else:
        logger.info("init weights.")

    best_loss = 2 ^ 16
    for epoch in range(epoch, cfg.max_epoch):
        # train
        train_one_epoch(Encoder=Encoder,
                        Decoder=Decoder,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        lpips=lpips_metric,
                        data_loader=train_loader,
                        epoch=epoch,
                        tb_writer=tb_writer,
                        cfg=cfg)
        # validate
        val_result = evaluate_one_epoch(Encoder=Encoder,
                                        Decoder=Decoder,
                                        lpips=lpips_metric,
                                        data_loader=val_loader,
                                        epoch=epoch,
                                        tb_writer=tb_writer,
                                        cfg=cfg)

        # tensorboard
        for k, v in val_result.items():
            tb_writer.add_scalars(f"data/{k}", {"val": v}, global_step=(epoch + 1) * cfg.iter_per_epoch)
        logger.info(f"validation: current epoch:[{epoch}/{cfg.max_epoch - 1}] - "
                    f"loss:{round(val_result['loss'], 4)} "
                    f"img_loss:{round(val_result['img_loss'], 4)} "
                    f"msg_loss:{round(val_result['msg_loss'], 4)} "
                    f"bit_acc:{round(val_result['bit_acc'], 4)} "
                    f"str_acc:{round(val_result['str_acc'], 2)} ")
        # save weights
        if val_result["loss"] < best_loss and epoch >= cfg.start_save_best:  # todo 不合理
            save_state(Encoder, Decoder, optimizer, scheduler, epoch + 1, cfg, "best.pth")
        save_state(Encoder, Decoder, optimizer, scheduler, epoch + 1, cfg, f"latest-{epoch}.pth")


def save_state(Encoder, Decoder, optimizer, scheduler, epoch, cfg, file_name):
    torch.save({"Encoder": Encoder.state_dict(),
                "Decoder": Decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, }, os.path.join(cfg.save_dir, cfg.exp_name, file_name))


if __name__ == '__main__':
    main()
