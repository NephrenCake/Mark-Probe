from tools.interface.attack import *

img_path = "test_source\\01.jpg"
image = cv2.imread(img_path)


def cv_show(img, name="default"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_brightness_trans(brightness):
    _img = brightness_trans(image, brightness)
    cv_show(np.hstack((image, _img)), "brightness demo brightness={}".format(brightness))


def test_contrast_trans(contrast_factor):
    _img = contrast_trans(image, contrast_factor)
    cv_show(np.hstack((image, _img)), f"contrast adjust contrast_factor={contrast_factor}")


def test_saturation_trans(saturation_factor):
    _img = saturation_trans(image, saturation_factor)
    cv_show(np.hstack((image, _img)), f"saturation adjust saturation_factor={saturation_factor}")


def test_gaussian_blur(flag):
    _img = gaussian_blur(image, flag)
    cv_show(np.hstack((image, _img)), "Gaussian_blur")


def test_rand_noise(std, mean):
    _img = rand_noise(image, std, mean)
    cv_show(np.hstack((image, _img)), "rand_noise")
    pass


def test_jpeg_trans(factor):
    _img = jpeg_trans(image, factor)
    cv_show(np.hstack((image, _img)), "jpeg_trans")


def test_rand_erase(cover_rate):
    _img = rand_erase(image, cover_rate)
    cv_show(np.hstack((image, _img)), "rand_erase")


def test_grayscale_trans(flag):
    _img = grayscale_trans(flag)
    cv_show(np.hstack((image, _img)), "grayscale_trans")


if __name__ == "__main__":
    test_brightness_trans(0.3)
    test_contrast_trans(0.3)
    test_saturation_trans(0.5)
    test_gaussian_blur(True)
    test_rand_noise([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])
    test_jpeg_trans(30)
    rand_erase(0.1)
    test_grayscale_trans(True)
