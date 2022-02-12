import random
from PIL import Image, ImageEnhance, ImageOps
import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as transform_f

# 是否启用随机镜像变换
random_mirror = True


# shearX 变换 缩角度===================================================== 废弃
class ShearX(object):
    def __call__(self, img, magnitude):
        k = -1 if random.randint(-1,1)<0 else 1
        return transform_f.affine(img, angle=0, translate=(0, 0), scale=1, shear=k*magnitude*100)


# def ShearX(img, v):  # [-0.3, 0.3]
#     assert -0.3 <= v <= 0.3
#     if random_mirror and random.random() > 0.5:
#         v = -v
#     return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

class ShearY(object):
    def __call__(self, img, magnitude):
        k = -1 if random.randint(-1, 1) < 0 else 1
        return transform_f.affine(img, angle=0, translate=(0, 0), scale=1, shear=(0, k*magnitude*100))


# translate 平移变换====================================================
class TranslateX(object):
    def __call__(self, img, magnitude):
        return transforms.RandomAffine(degrees=0, translate=(magnitude,0))(img)


class TranslateY(object):
    def __call__(self, img, magnitude):
        return transforms.RandomAffine(degrees=0, translate=(0,magnitude))(img)


# 旋转========================================================================
class Rotate(object):
    def __call__(self, img, magnitude):
        return transforms.RandomAffine(degrees=(0,magnitude))(img)


# 增加对比度  最大化对比度========================================================
class AutoContrast(object):
    def __call__(self, img, _):
        return transform_f.autocontrast(img)


# 随机定义对比度==================================================================
class Contrast(object):
    def __call__(self, img, magnitude):
        return transform_f.adjust_contrast(img=img, contrast_factor=magnitude)
        # return transforms.ColorJitter(contrast=(0,magnitude))(img)


# 翻转颜色=====================================================================
class Invert(object):
    def __call__(self, img, magnitude):
        return transform_f.invert(img);


# 均衡化======================================================================
class Equalize(object):
    def __call__(self, img, magnitude):
        '''
        由于transforms.functional 中的equalize只接受torch.int8 类型的值
        计算图中传入的参数又是 torch.float 并且是被归一化到0-1之间的值
        # 官方解释了 改变tensor的dtype是不会打断backward 但是 精度的损失会不会我就不太清楚了 主要是精度的损失有些大
        '''
        img = img*255
        if len(img.shape)==4:
            for i in range(img.shape[0]):
                img[i] = transform_f.equalize(img.byte()).float()
            return img/255.0
        elif len(img.shape)==3:
            return transform_f.equalize(img.byte()).float()/255.0


# 图像翻转=====================================================================
class Flip():  # not from the paper
    def __call__(self, img, _):
        k = 1 if random.randint(-1,1)>0 else -1
        if k==1:
            return transform_f.hflip(img)
        else:
            return transform_f.vflip(img)


# Solarize v_so[0, 256] 指定一个像素值，对原图中高于该值的像素进行翻转操作，简单来说是一种部分invert操作
class Solarize(object):
    def __call__(self, img, magnitude):
        return transform_f.solarize(img=img,threshold=magnitude/255)

# 减少组成图片的色彩的种类=========================================================
class Posterize(object):
    def __call__(self, img, magnitude):
        img = img*255
        img_ = img.byte()
        return transform_f.posterize(img=img_,bits=magnitude).float()

# 调整色彩平衡，简单来说可以理解为调整图像色调和饱和度。=====================================
class Color(object):
    def __call__(self, img, magnitude):
        return transform_f.adjust_saturation(img=img, saturation_factor=magnitude)

# 亮度==============================================================================
class Brightness(object):
    def __call__(self, img, magnitude):
        return transform_f.adjust_brightness(img=img,brightness_factor=magnitude*10)
        # return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))


# 锐化===============================================================================
class Sharpness(object):
    def __call__(self, img, magnitude):
        return transform_f.adjust_sharpness(img=img, sharpness_factor=magnitude * 10)



# 随机块遮挡===========================================================================

class Cutout(object):

    def __call__(self, img, magnitude):
        assert 0.0 <= magnitude <= 0.2
        if magnitude <= 0.:
            return img

        magnitude = magnitude * img.size[0]
        return self.CutoutAbs(img, magnitude)

    def CutoutAbs(self, img, magnitude):
        # assert 0 <= v <= 20
        if magnitude < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - magnitude / 2.))
        y0 = int(max(0, y0 - magnitude / 2.))
        x1 = min(w, x0 + magnitude)
        y1 = min(h, y0 + magnitude)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

# 图片叠加==============================================================================
def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f
