import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    print(rows, cols)
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def reorder(my_points):
    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)

    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]

    return my_points_new


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src


def max_contour_idx(contours):
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    return max_idx


def max_contour_order(contours, num):
    big_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:num]
    return big_contour


def img_find(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(255 - th2, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    # max_idx = max_contour_idx(contours)  # 返回最大轮廓的index
    biggest = quadrangular_fit(contours)
    return biggest


def quadrangular_fitting(max_contour):
    for factor in range(1, 20, 1):
        factor = factor / 100  # 0.002 ~ 0.2
        peri = cv2.arcLength(max_contour, True) * factor
        approx = cv2.approxPolyDP(max_contour, peri, True)
        if len(approx) == 4:
            biggest = reorder(approx)
            return biggest
        else:
            return -1


def quadrangular_fit(contour):
    contours = sorted(contour, key=cv2.contourArea, reverse=True)[:2]
    for c in contours:
        for factor in range(1, 20, 1):
            factor = factor / 100  # 0.002 ~ 0.2
            peri = cv2.arcLength(c, True) * factor
            approx = cv2.approxPolyDP(c, peri, True)
            if len(approx) == 4:
                biggest = reorder(approx)
                return biggest
    return -1


def canny_find(img_gray):
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(img_gray, contours, -1, (255, 255, 0), 4)  # DRAW ALL DETECTED CONTOURS
    # max_idx = max_contour_idx(contours)  # 返回最大轮廓的index
    biggest = quadrangular_fit(contours)
    return biggest


def perspective_correction(biggest, img):
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [448, 0], [0, 448], [448, 448]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (448, 448))
    return img_warp


def pre_treat(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    img_threshold = cv2.Canny(img_blur, 200, 200)  # APPLY CANNY BLUR
    kernel = np.ones((2, 2))
    img_dilate = cv2.dilate(img_threshold, kernel, iterations=2)  # APPLY DILATION
    img_threshold = cv2.erode(img_dilate, kernel, iterations=1)  # APPLY EROSION
    return img_threshold


def data_package(data_list, num):
    res_list = []
    for i in range(len(data_list)):

        if type(data_list[i][0]) != int:
            point1 = data_list[i][0][0][0]
            point2 = data_list[i][0][1][0]
            point3 = data_list[i][0][2][0]
            point4 = data_list[i][0][3][0]
            point = [{'id': 1, 'x': point1[0], 'y': point1[1]},
                     {'id': 2, 'x': point2[0], 'y': point2[1]},
                     {'id': 3, 'x': point3[0], 'y': point3[1]},
                     {'id': 4, 'x': point4[0], 'y': point4[1]},
                     # {'img': data_list[i][1]}
                     {'all_point': data_list[i][0]}
                     ]
            # data_list[i][1]
            res_list.append(point)
        else:
            res_list.append([-1])
    return res_list
