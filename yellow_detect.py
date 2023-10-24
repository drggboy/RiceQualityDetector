import cv2
import obj_detect
import numpy as np

def on_color_change(param):
    pass

'''
通过设置HSV颜色参数选取指定的颜色
HSV颜色相关知识请参考：https://www.cnblogs.com/wangyblzu/p/5710715.html
'''
def detect_yellow_spot(im):
    im = cv2.resize(im, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    # im = cv2.imread('rice/yellow.jpg', cv2.IMREAD_COLOR)
    # cv2.imshow('im',im)
    # cv2.waitKey(0)
    # 计算大米总面积
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    roi_area = cv2.countNonZero(gray)
    #cv2.namedWindow('image')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image',500,250)
    cv2.moveWindow('image',0,0)
    # 创建颜色变化的轨迹栏
    cv2.createTrackbar('Hmin','image',0,180,on_color_change)
    cv2.createTrackbar('Hmax','image',60,180,on_color_change)
    cv2.createTrackbar('Smin','image',10,255,on_color_change)
    cv2.createTrackbar('Smax','image',255,255,on_color_change)
    cv2.createTrackbar('Vmin','image',46,255,on_color_change)
    cv2.createTrackbar('Vmax','image',255,255,on_color_change)

    while True:
        hmin = cv2.getTrackbarPos('Hmin','image')
        hmax = cv2.getTrackbarPos('Hmax','image')
        smin = cv2.getTrackbarPos('Smin','image')
        smax = cv2.getTrackbarPos('Smax','image')
        vmin = cv2.getTrackbarPos('Vmin','image')
        vmax = cv2.getTrackbarPos('Vmax','image')

        lower=np.array([hmin,smin,vmin])
        upper=np.array([hmax,smax,vmax])
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower,upper)
        im1 = im.copy()
        im1[mask > 0] = [255, 0, 0]
        # cv2.imshow('white spot', im1)
        # im1 = cv2.bitwise_and(im,im,mask=mask)
        cv2.imshow('image2',im1)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 13:  # 按下回车键
            yellow_area = cv2.countNonZero(mask)
            now_yellow = yellow_area / roi_area
            print('now_yellow:', now_yellow)
        elif ch == 27:  # 按下Esc键
            break


if __name__ == "__main__":
    im_path = r'rice/whitespot.jpg'
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    # 检测大米轮廓，测量尺寸
    im2 = obj_detect.detect_rice(im)
    # 黄米检测
    detect_yellow_spot(im2)
