import os
import sys

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from tools.interface.attack import *


def cv_show(img, name="default", contrast=None):
    if contrast is not None:
        img = np.hstack((contrast, img))

    cv2.imshow(name, img)
    cv2.imwrite("out/" + name + ".jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_brightness_trans(img, brightness):
    _img = brightness_trans(img, brightness)
    cv_show(_img, f"brightness demo brightness={brightness}")
    return _img


def test_contrast_trans(img, contrast_factor):
    _img = contrast_trans(img, contrast_factor)
    cv_show(_img, f"contrast adjust contrast_factor={contrast_factor}")
    return _img


def test_saturation_trans(img, saturation_factor):
    _img = saturation_trans(img, saturation_factor)
    cv_show(_img, f"saturation adjust saturation_factor={saturation_factor}")
    return _img


def test_hue_trans(img, hue_factor):
    _img = saturation_trans(img, hue_factor)
    cv_show(_img, f"hue adjust hue_factor={hue_factor}")
    return _img


def test_gaussian_blur(img, flag):
    _img = gaussian_blur(img, flag)
    cv_show(_img, "Gaussian_blur")
    return _img


def test_rand_noise(img, std, mean):
    _img = rand_noise(img, std, mean)
    cv_show(_img, "rand_noise")
    return _img


def test_jpeg_trans(img, factor):
    _img = jpeg_trans(img, factor)
    cv_show(_img, "jpeg_trans")
    return _img


def test_rand_erase(img, cover_rate):
    _img = rand_erase(img, cover_rate)
    cv_show(_img, "rand_erase")
    return _img


def test_grayscale_trans(img, flag):
    _img = grayscale_trans(img, flag)
    cv_show(_img, "grayscale_trans")
    return _img


def test_motion_blur(img, kernel_size):
    _img = motion_blur(img, kernel_size)
    cv_show(_img, f"motion_blur kernel_size{kernel_size}")
    return _img


if __name__ == "__main__":
    img_path = "test_source/COCO_train2014_000000001497.jpg"
    image = cv2.imread(img_path)
    image = cv2.resize(image, (448, 448))

    image = test_brightness_trans(image, 0.3)
    image = test_contrast_trans(image, 0.5)
    image = test_saturation_trans(image, 1)
    image = test_hue_trans(image, 0.9)
    # image = test_gaussian_blur(True)
    image = test_motion_blur(image, 3)
    image = test_rand_noise(image, [0.229, 0.224, 0.225], [0.485, 0.456, 0.406])
    image = test_jpeg_trans(image, 60)
    image = test_grayscale_trans(image, True)

    image = test_rand_erase(image, 0.2)

