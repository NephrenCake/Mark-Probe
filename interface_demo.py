import cv2

img_path = "D:\learning\COCOTrain+Val\\test2014\COCO_test2014_000000000001.jpg"
image = cv2.imread(img_path)

'''
所有的测试用例显示的图片中 左侧是原图。
'''

def cv_show(img, name = "default"):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def brightness_trans_demo(brightness):
    _img = brightness_trans(image, brightness)
    cv_show(np.hstack((image,_img)),"brightness demo brightness={}".format(brightness))

def contrast_trans_demo(contrast_factor):
    _img = contrast_trans(image, contrast_factor)
    cv_show(np.hstack((image,_img)),"contrast adjust contrast_factor={}"/format(contrast_factor))

def saturation_trans_demo(saturation_factor):
    _img = saturation_trans(image, saturation_factor)
    cv_show(np.hstack((image, _img)), "saturation adjust saturation_factor={}" / format(saturation_factor))

def gaussian_blur_demo(flag):
    _img = gaussian_blur(image,flag)
    cv_show(np.hstack((image, _img)), "Gaussian_blur")

def rand_noise_demo(std,mean):
    _img = rand_noise(image,std,mean)
    cv_show(np.hstack((image, _img)), "rand_noise")
    pass

def jpeg_trans_demo(factor):
    _img = jpeg_trans(image, factor)
    cv_show(np.hstack((image, _img)), "jpeg_trans")

def rand_erase_demo(cover_rate):
    _img = rand_erase(image,cover_rate)
    cv_show(np.hstack((image, _img)), "rand_erase")


def grayscale_trans(flag):
    _img = grayscale_trans(flag)
    cv_show(np.hstack((image, _img)), "grayscale_trans")
    pass

def main():
    brightness_trans_demo(0.3)
    contrast_trans_demo(0.3)
    saturation_trans_demo(0.5)
    gaussian_blur_demo(True)
    rand_noise_demo([0.229, 0.224, 0.225],[0.485, 0.456, 0.406])
    jpeg_trans_demo(30)
    rand_erase(0.1)
    grayscale_trans(True)
    pass

if __name__=="__main__":
    main()

