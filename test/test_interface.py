import numpy as np
import torch

import cv2


img_path = "D:\learning\COCOTrain+Val\\test2014\COCO_test2014_000000000001.jpg"
# img_path = "D:\learning\pythonProjects\HiddenWatermark1\\test\\test_source\\01.jpg"
image = cv2.imread(img_path)

def cv_show(img, name = "default"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_brightness_trans():
    '''
    显示 变换之后的图片和原图的差距
    image: 原图
    img: 变换之后的图片
    '''
    img = brightness_trans(image, 0.3)
    cv_show(np.hstack((image, img)))

def test_contrast_trans():
    img = contrast_trans(image,0.3)
    cv_show(np.hstack((image,img)))

def test_saturation_trans():
    img = saturation_trans(image,0.5)
    cv_show(np.hstack((image, img)))

def test_hue_trans():
    img = hue_trans(image, 0.1)
    cv_show(np.hstack((image, img)))

def test_gaussian_blur():
    img = gaussian_blur(image, True)
    cv_show(np.hstack((image, img)))

def test_rand_noise():
    img = rand_noise(image,[0.1,0.2,0.1])
    cv_show(np.hstack((image, img)), "Gaussian_blur")
    pass

def test_jpeg_trans():
    # 对img增加一个维度： np.expand_dims(img,0)添加一个维度
    # 如何reshape：opencv img.transpose(2,0,1) c h w
    # 在使用jpeg压缩时候出现了图片和原图的不一致
    factor = 30
    im = image

    img = jpeg_trans(im, factor)
    # img2 = jpeg_trans(image, 30)
    cv_show(np.hstack((im, img)), "Jpeg Compression  factor{}".format(factor))
    print(img[10][10])
    print(image[10][10])

def test_grayscale_trans():
    print(len(image.shape))
    img = grayscale_trans(image,True)
    print(img.shape)
    print(img[0][0])
    cv_show(np.hstack((image, img)), "grayscale_trans")

def test_rand_erase():
    img_ = rand_erase(image, 0.4, 20)
    cv_show(np.hstack((image, img_)), "grayscale_trans")
    pass

def test():
    a = torch.empty(1).uniform_(0, 1+0.3)
    print(a)
    pass


if __name__=="__main__":
    # test_brightness_trans()
    # test_contrast_trans()
    # test_saturation_trans()
    # test_hue_trans()
    # test_gaussian_blur()
    # test_rand_noise()
    # test_jpeg_trans()
    # test_grayscale_trans()
    # test_rand_erase()
    test()
