import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from steganography.utils.distortion import rand_crop

img_path = "test_source/COCO_train2014_000000000009.jpg"
img_size = (448, 448)
msg_size = 96
scale = {
    "angle_trans": 30,
    "cut_trans": 0.5,
    "perspective_trans": 0.1,
}

img = transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor()
])(Image.open(img_path).convert("RGB")).unsqueeze(0)
torchvision.utils.save_image(img, "test_source/test.jpg")


def test_crop():
    global img

    img = rand_crop(img, scale, change_pos=False)
    torchvision.utils.save_image(img, "test_source/test_crop.jpg")


def test_perspective():
    def get_params(width: int, height: int, distortion_scale: float):
        distort_width = int(distortion_scale * (width // 2)) + 1
        distort_height = int(distortion_scale * (height // 2)) + 1

        topleft = [
            int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
            int(torch.randint(-distort_height, distort_height, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
            int(torch.randint(-distort_height, distort_height, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - distort_width, width + distort_width, size=(1, )).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(-distort_width, distort_width, size=(1, )).item()),
            int(torch.randint(height - distort_height, height + distort_height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    global img

    startpoints, endpoints = get_params(img_size[0], img_size[0], scale["perspective_trans"])
    img = F.perspective(img, startpoints, endpoints)
    torchvision.utils.save_image(img, "test_source/test_perspective.jpg")


if __name__ == '__main__':
    test_crop()
    test_perspective()
