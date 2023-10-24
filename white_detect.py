import cv2
import matplotlib.pyplot as plt
import obj_detect

'''
trackbar值变动回调函数
'''
def on_threshold_change(param):
    pass

'''
通过设定灰度阈值方式检测白点
'''
def detect_white_spot(im):
    # im = cv2.imread('rice/whitespot.jpg', cv2.IMREAD_COLOR)
    # im = im2
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # 绘制感兴趣区域的直方图
    mask = gray.copy()
    mask[gray>0] = 255;    #掩膜
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 255])
    plt.plot(hist)
    plt.show()

    roi_area = cv2.countNonZero(gray)     #计算大米总面积
    # print('roi_area:', roi_area)

    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)

    cv2.namedWindow('white spot')
    cv2.createTrackbar('Threshold','white spot',190,255,on_threshold_change)
    while True:
        thres_val = cv2.getTrackbarPos('Threshold','white spot')
        ret,thres = cv2.threshold(gray, thres_val, 255, cv2.THRESH_BINARY)    #返回的ret表示阈值
        # 计算当前垩白度
        # cv2.imshow('thres',thres)
        # cv2.waitKey(0)
        im1 = im.copy()
        im1[thres>0] = [255,0,0]
        cv2.imshow('white spot',im1)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 13:   # 按下回车键
            white_area = cv2.countNonZero(thres)
            now_Chalk_whiteness = white_area / roi_area
            print('now_Chalk_whiteness:', now_Chalk_whiteness)
        elif ch == 27:         # 按下Esc键
            break

if __name__ == "__main__":
    im_path = r'rice/whitespot.jpg'
    im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    # 检测大米轮廓，测量尺寸
    im2 = obj_detect.detect_rice(im)
    # 白点检测
    detect_white_spot(im2)
