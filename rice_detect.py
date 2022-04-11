import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
使用findContours方法查找对象轮廓
'''
def detect_objects(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0) #通过高斯滤镜过滤高频噪音
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #查找阈值
    #im1, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    #注释第12行，对比找到的轮廓树，会发现高斯滤镜滤掉了许多噪音，提升了效率
    print('查找到',len(contours),'个轮廓')
    objects = []
    painted = []

    # 寻找物体的凸包并绘制凸包的轮廓
    for i in range(len(contours)):
        cur_index = len(contours) -1 - i #从最里层的轮廓开始绘制
        cnt = contours[cur_index]
        hull = cv2.convexHull(cnt) #查找轮廓的凸包多边形
        length = len(hull)
        # 过滤凸包点数小于20的轮廓，进一步去除噪音
        if length > 20:
            if cur_index in painted:
                continue
            painted.append(hierarchy[0][cur_index][3]) #把当前绘制的轮廓上级轮廓放入已绘制列表，避免重复绘制
            rect = cv2.minAreaRect(hull) #最小外界矩形，用于求计算位置和长宽
            box = np.int0(cv2.boxPoints(rect)) #外界矩形的坐标点
            painted.append(cur_index)
            objects.append({
                'box':box,
                'rect':rect,
                'hull':hull
            })

    return objects


#检测大米轮廓，测量尺寸
def detect_rice():
    im = cv2.imread('rice/rice.jpg', cv2.IMREAD_COLOR)
    objs = detect_objects(im)
    for obj in objs:

        #绘制目标最小外界矩形
        cv2.drawContours(im, [obj.get('box')], 0, (0,255,0), 3)

        #绘制目标长度和宽度数据
        rect = obj.get('rect')
        w,h = rect[1]
        w = int(w)
        h = int(h)
        x,y = rect[0]
        x = int(x)
        y = int(y)
        cv2.putText(im,'width:%d,height:%d'%(w,h),(x+50,y),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,0,255),1)

        #在目标凸包轮廓上绘制半透明的蓝色
        im_hull = np.zeros((im.shape),dtype=np.uint8)
        cv2.fillConvexPoly(im_hull,obj.get('hull'),(255,0,0,127))
        im = cv2.addWeighted(im,1,im_hull,0.5,0)

    cv2.imshow('image',im)
    cv2.waitKey(0)




'''
trackbar值变动回调函数
'''
def on_threshold_change(param):
    pass

'''
通过设定灰度阈值方式检测白点
'''
def detect_white_spot():
    im = cv2.imread('rice/whitespot.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('white spot')
    cv2.createTrackbar('Threshold','white spot',190,255,on_threshold_change)
    while True:
        thres_val = cv2.getTrackbarPos('Threshold','white spot')
        ret,thres = cv2.threshold(gray, thres_val, 255, cv2.THRESH_BINARY)
        im1 = im.copy()
        im1[thres>0] = [255,0,0]
        cv2.imshow('white spot',im1)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break

def on_color_change(param):
    pass


'''
通过设置HSV颜色参数选取指定的颜色
HSV颜色相关知识请参考：https://www.cnblogs.com/wangyblzu/p/5710715.html
'''
def detect_yellow_spot():
    im = cv2.imread('rice/yellow.jpg', cv2.IMREAD_COLOR)
    cv2.namedWindow('image')
    # 创建颜色变化的轨迹栏
    cv2.createTrackbar('Hmin','image',26,360,on_color_change)
    cv2.createTrackbar('Hmax','image',34,360,on_color_change)
    cv2.createTrackbar('Smin','image',43,255,on_color_change)
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
        im1 = cv2.bitwise_and(im,im,mask=mask)
        cv2.imshow('image',im1)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break

'''
通过梯度方式检测米粒内部断层
本例采用Scharr算子
关于Sobel和Scharr算子，请参阅：https://www.cnblogs.com/yibeimingyue/p/10878514.html
'''
def detect_fracture():
    img = cv2.imread(r'rice\fracture.jpg',cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    grad_x = cv2.Scharr(img,cv2.CV_32F,1,0) # 采用CV_32F是怕数据被截断 ，对x求导
    grad_y = cv2.Scharr(img,cv2.CV_32F,0,1) # 0，1，2->0表示这个方向没有进行求导，对y求导

    gradx = cv2.convertScaleAbs(grad_x) #X方向梯度
    grady = cv2.convertScaleAbs(grad_y) #Y方向梯度
    gradxy = cv2.addWeighted(gradx,0.5,grady,0.5,0) #加权合并X,Y方向梯度

    plt.subplot(2,2,1),plt.imshow(img)
    plt.title('Image')
    plt.subplot(2,2,2),plt.imshow(gradx)
    plt.title('Gradient X')
    plt.subplot(2,2,3),plt.imshow(grady)
    plt.title('Gradient Y')
    plt.subplot(2,2,4),plt.imshow(gradxy)
    plt.title('Gradient XY')
    plt.show()

detect_rice()
#detect_yellow_spot()
#detect_white_spot()
#detect_fracture()


cv2.destroyAllWindows()