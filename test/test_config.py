from steganography.config.config import TrainConfig

cfg = TrainConfig()
cfg.set_iteration(10)
for epoch in range(0, 10):
    for iters in range(0, 10):
        scale = cfg.get_cur_scales(cur_iter=iters, cur_epoch=epoch)
        print("epoch:", epoch)
        # print("erasing_trans", scale["erasing_trans"])
        # print("perspective", scale["perspective_trans"])
        print("motion_blur",round(scale["motion_blur"]))

