import numpy as np
import  cv2
# 灰度化方法
def max_grayscale(img):    #最大值灰度化
    h,w = img.shape[0:2]
    gray = np.zeros((h, w), dtype=img.dtype)  # 最大值
    for i in range(h):
        for j in range(w):
            gray[i, j] = max(img[i, j, 0], img[i, j, 1], img[i, j, 2])  # 最大值
    return gray

def H_grayscale(img):      #HSV中H分量
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h = img_hsv[..., 0]
    gray = img_h
    return gray

def Hist_grayscale(img):    #对比度拉伸
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img
    # img = max_grayscale(img)
    Imax = np.max(img)
    Imin = np.min(img)
    Max = 255
    Min = 0
    im_cs = (img - Imin) / (Imax - Imin) * (Max - Min) + Min
    im_cs = im_cs.astype("uint8")
    return im_cs

def eH_grayscale(img):  #equalizeHist  直方图均衡化
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # img_gray = max_grayscale(img)
    eH_img = cv2.equalizeHist(img_gray)
    return eH_img

def clahe_grayscale(img):  #自适应直方图均衡化
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    # img_gray = max_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    clahe = clahe.apply(img_gray)
    return clahe