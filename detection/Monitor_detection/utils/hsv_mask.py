import cv2
import numpy as np


def paper_range(frame):
    # img = cv2.GaussianBlur(frame, (5, 5), 0)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[2], channels[2])

    cv2.merge(channels, img)

    white_lower = np.array([0, 0, 120], np.uint8)

    white_upper = np.array([180, 36, 255], np.uint8)

    white = cv2.inRange(img, white_lower, white_upper)

    return white


def monitor_hsv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[2], channels[2])

    cv2.merge(channels, img)

    black_lower = np.array([0, 0, 0], np.uint8)

    black_upper = np.array([180, 255, 46], np.uint8)

    black = cv2.inRange(img, black_lower, black_upper)
    return black
