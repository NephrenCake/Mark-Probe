import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from steganography.utils.distortion import rand_crop

img_path = "out/jpeg_trans.jpg"
img_size = (448, 448)
msg_size = 96
scale = {
    "angle_trans": 30,
    "cut_trans": 0.7,
    "perspective_trans": 0.1,
}

img = transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor()
])(Image.open(img_path).convert("RGB")).unsqueeze(0)


def show_result(img0: torch.Tensor, save_img=None, show_img=True):
    """
    展示一张tensor形式的图片，并可选保存图片。若为4维度，则自动解除第一个维度
    """
    img_ = img0.clone().detach()
    if len(img_.size()) == 4:
        assert img_.size()[0] == 1, "请勿放多张图片"
        img_ = img_.squeeze(0)

    if save_img is not None:
        assert isinstance(save_img, str), "请给出字符串形式文件名"
        assert save_img.endswith((".jpg", ".png")), "请将文件名以 .jpg 或 .png 结尾"
        torchvision.utils.save_image(img_, save_img)

    if show_img:
        img_ = transforms.ToPILImage()(img_)
        img_.show()


def test_crop():
    global img
    img_ = img.clone().detach()
    # img_ = transforms.RandomSizedCrop((224, 224))(img_)
    img_ = rand_crop(img_, scale, change_pos=False)
    show_result(img_, "out/crop.jpg")


def test_perspective():
    def get_params(width: int, height: int, distortion_scale: float):
        distort_width = int(distortion_scale * (width // 2)) + 1
        distort_height = int(distortion_scale * (height // 2)) + 1

        topleft = [
            int(torch.randint(-distort_width, distort_width, size=(1,)).item()),
            int(torch.randint(-distort_height, distort_height, size=(1,)).item())
        ]
        topright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1,)).item()),
            int(torch.randint(-distort_height, distort_height, size=(1,)).item())
        ]
        botright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1,)).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1,)).item())
        ]
        botleft = [
            int(torch.randint(-distort_width, distort_width, size=(1,)).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1,)).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    global img
    img_ = img.clone().detach()

    startpoints, endpoints = get_params(img_size[0], img_size[0], scale["perspective_trans"])
    img_ = F.perspective(img_, startpoints, endpoints)
    show_result(img_, "out/perspective.jpg")


if __name__ == '__main__':
    test_crop()
    test_perspective()
