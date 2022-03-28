import cv2
import numpy as np
from PIL import Image
from detection.Monitor_detection.utils import Mask_seg, box, Monitor_detect


def predict(img, model, thresold_value=38):
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
    # -------------------------------------------------------------------------#

    if isinstance(img, Image.Image):
        img = np.array(img)

    # 格式转变，BGRtoRGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origin_img = img.copy()
    # 转变成Image
    img = Image.fromarray(np.uint8(img))

    # 进行检测
    single_image, image = model.detect_image(img)  # 此处image是原图和分割图混合完成后的图

    single_image = np.array(single_image)

    image = Mask_seg.mask_cut(origin_img, single_image)  # 此处image是分割完后的图
    image_box, mark_img = box.yuchuli(origin_img, image, thresold_value=thresold_value)
    image_box_, point = Monitor_detect.Monitor_find(origin_img, image_box)
    if type(point) == int:
        image_box_ = mark_img
        frame = np.array(image_box_)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return -1

    frame = np.array(image_box_)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    point1 = point[0][0]
    point2 = point[1][0]
    point3 = point[2][0]
    point4 = point[3][0]
    return [point1, point2, point3, point4, frame]
