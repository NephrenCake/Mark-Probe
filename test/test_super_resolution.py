import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms

from tools.utils.bch_utils import BCHHelper
from steganography.utils.SuperResolution.get_sr import super_resolution

from steganography.models.MPNet import MPDecoder, MPEncoder

'''
对 图像 encoded之后的res_img 进行超分在叠加到 原图上
'''

origin_img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\\test_source\\test_superResolution.jpg"
pretrained = "D:\learning\pythonProjects\HiddenWatermark1\steganography\\train_log\CI-test_2022-02-19-13-23-41\\best.pth"  # 使用的模型： CI-test_2022-02-19-13-23-41
uid_test = 114514

msg_size = 96
img_size = (448, 448)
device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu"


# todo 一些性能方面的优化
# todo 最好能够直接将 超分模型嵌入到encode里面 然后encoder 就能直接返回原图大小的图片了。
def encode(img_path, uid):
    """
    根据 img_path 对图像进行 encode
    return  res_img 残差图, 叠加图像原图大小  PIL格式
    """

    bch = BCHHelper()
    Encoder = MPEncoder().to(device)
    checkpoint = torch.load(pretrained, map_location=device)
    Encoder.load_state_dict(checkpoint['Encoder'])
    Encoder.eval()
    img = Image.open(img_path).convert("RGB")
    h, w = img.size
    img_ = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])(img).to(device)
    img = transforms.ToTensor()(img)
    dat, now, key = bch.convert_uid_to_data(uid)
    msg = torch.tensor(bch.encode_data(dat), dtype=torch.float)
    res_img = Encoder({"img": img_.unsqueeze(0), "msg": msg.unsqueeze(0)})
    max_vr = torch.max(res_img)
    min_vr = torch.min(res_img)
    # 现在出现的问题是
    res_max = super_resolution(res_img)
    res_max = transforms.Resize((w, h))(res_max).clamp(0., max_vr.item())
    img = img + res_max
    res_img_PIL = transforms.ToPILImage()(res_max + 0.5)
    img_PIL = transforms.ToPILImage()(img.squeeze(0)).show()
    # img_PIL.save("D:\learning\pythonProjects\HiddenWatermark1\\test\\test_result/encoded.png")
    # res_img_PIL.save("D:\learning\pythonProjects\HiddenWatermark1\\test\\test_result/res.png")
    return res_img_PIL


def step2():
    img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\\test_result\\res.png"
    img = cv2.imread(origin_img_path)
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    res_img = cv2.imread(img_path)
    h, w, c = img.shape
    res_img = cv2.resize(res_img, (w, h))

    # 不能直接叠加
    img = img + res_img

    # cv2.imshow("res",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #

    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # 对这个图像进行超分  3.421875  四倍放大
    # D:\learning\pythonProjects\HiddenWatermark1\steganography\utils\SuperResolution\out\res.png 获得了放大四倍之后的 res 图像


def main():
    encode(img_path=origin_img_path, uid=uid_test)  # 这一步先将 高清图片编码 将res 图保存


if __name__ == "__main__":
    main()
