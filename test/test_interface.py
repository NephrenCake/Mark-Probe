import numpy as np

from steganography.utils.interface.distortion_interface import *
import cv2


img_path = "D:\learning\COCOTrain+Val\\test2014\COCO_test2014_000000000001.jpg"
image = cv2.imread(img_path)

def cv_show(img, name = "default"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_brightness_trans():

    img = brightness_trans(image, 0.4)
    cv_show(np.hstack((img,image)))


if __name__=="__main__":
    test_brightness_trans()