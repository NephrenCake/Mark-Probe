import cv2
import numpy as np

'''

'''

def brightness_trans(img, brightness):
    '''
    传入的是cv格式的图片
    对图片进行 亮度调整 将cv图片中的每一个像素的灰度值增加1+brightness% 的量
    '''

    # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape

    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, brightness+1, blank, 0, 0)
    return dst
