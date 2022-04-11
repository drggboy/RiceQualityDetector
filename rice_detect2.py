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
    # cv2.imshow('gray',gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #查找阈值
    cv2.imshow('thresh',thresh)
    # 对二值化图片降噪
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
    # thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2)  # 开运算
    # cv2.imshow('thresh_open', thresh_open)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 查找轮廓
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt1,cnt2=maxAndSubMax(contours)
    print('max=',hull_length(cnt1))
    print('submax=',hull_length(cnt2))
    # print('len(contours[3])',len(contours[3]))
    # cv2.imshow('1',im)
    # cv2.drawContours(im, contours[3],-1, (0, 0, 255), 3)  ####
    # cv2.imshow('test_contours', im)
    # hull_length(contours[3])
    #注释第12行，对比找到的轮廓树，会发现高斯滤镜滤掉了许多噪音，提升了效率
    # print('查找到',len(contours),'个轮廓')
    objects = []
    painted = []

    # 寻找物体的凸包并绘制凸包的轮廓
    k = 0 ####
    for i in range(len(contours)):
        cur_index = len(contours) -1 - i # 从最里层的轮廓开始绘制
        # cur_index = i
        cnt = contours[cur_index]
        hull = cv2.convexHull(cnt) #查找轮廓的凸包多边形
        length = len(hull)
        # 过滤凸包点数小于20的轮廓，进一步去除噪音
        if length > 20:
        # if cur_index == 3:
        #     print(cur_index)
            if cur_index in painted:
                continue
            k = k+1   ####
            # print(k)  #####
            # print('max_index=',cur_index)
            # cv2.drawContours(im, [contours[cur_index]], -1, (0, 0, 255),-1)
            # cv2.imshow('3',im)
            painted.append(hierarchy[0][cur_index][3]) #把当前绘制的轮廓的内嵌轮廓放入已绘制列表，避免重复绘制
            rect = cv2.minAreaRect(hull) #最小外接矩形，用于求计算位置和长宽
            box = np.int0(cv2.boxPoints(rect)) #外接矩形的坐标点
            # cv2.fillPoly(im, [contours[cur_index]], (255, 0, 0))
            # cv2.imshow('test',im)
            painted.append(cur_index)
            objects.append({
                'box':box,      #外接矩形的坐标点
                'rect':rect,    #最小外接矩形
                'hull':hull,     #轮廓的凸包多边形
                'cnt':cnt       #轮廓
            })
    print('满足条件的轮廓：',k)
    print('len(obj)',len(objects))
    return objects


#检测大米轮廓，测量尺寸
def detect_rice(im):
    im = cv2.imread(im, cv2.IMREAD_COLOR)
    #im = cv2.imread('rice/whitespot.jpg', cv2.IMREAD_COLOR)
    im2 = im.copy()

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

        # 在目标凸包轮廓上绘制半透明的蓝色
        im_hull = np.zeros((im.shape),dtype=np.uint8)
        cv2.fillConvexPoly(im_hull, obj.get('hull'), (255, 0, 0, 127))
        # cv2.imshow('im_hull',im_hull)
        im = cv2.addWeighted(im,1,im_hull,0.5,0)
        # cv2.imshow('im',im)
        # 提取出感兴趣区域
        im_hull_roi = np.zeros((im.shape), dtype=np.uint8)
        cv2.fillPoly(im_hull_roi, [obj.get('hull')], (255, 255, 255))
        cv2.imshow('im_hull_roi',im_hull_roi)

        # im_cnt_roi = np.zeros((im.shape), dtype=np.uint8)
        # cv2.fillPoly(im_cnt_roi, [obj.get('cnt')], (255, 255, 255))
        # cv2.imshow('im_cnt_roi',im_cnt_roi)
        # median = cv2.GaussianBlur(im_hull_roi, (5, 5), 1)}}}}        # cv2.imshow('median',median)

        # 进行膨胀、腐蚀、闭运算
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # kernel大小，准备进行开运算
        # dilated = cv2.dilate(im_hull_roi, kernel,5)
        # erode = cv2.erode(im_hull_roi, kernel, 100)
        # erode_next = cv2.dilate(erode, kernel, 100)
        # # open = cv2.morphologyEx(im_hull_roi, cv2.MORPH_CLOSE, kernel,1000)  # 开运算
        # cv2.imshow('ro',im_hull_roi)
        # cv2.imshow('dilated', dilated)
        # cv2.imshow('erode', erode_next)
        # cv2.imshow('test',open)
        # cv2.fillConvexPoly(im_hull_roi, obj.get('hull'), (255, 255, 255))
        im2 = cv2.bitwise_and(im2,im_hull_roi)
        # cv2.imwrite('rice_roi.jpg',im2)
        # cour = detect_objects(im2)

    cv2.imshow('detect_rice',im)
    cv2.waitKey(0)
    cv2.imshow('rice_roi', im2)
    cv2.waitKey(0)
    return im2




'''
trackbar值变动回调函数
'''
def on_threshold_change(param):
    pass

'''
通过设定灰度阈值方式检测白点
'''
def detect_white_spot(im2):
    # im = cv2.imread('rice/whitespot.jpg', cv2.IMREAD_COLOR)
    im = im2
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray',gray)
    cv2.waitKey(0)

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
    #cv2.namedWindow('image')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('image', 0)
    # 创建颜色变化的轨迹栏
    cv2.createTrackbar('Hmin','image',15,360,on_color_change)
    cv2.createTrackbar('Hmax','image',66,360,on_color_change)
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

def hull_length(cnt):     #用于计算轮廓凸包长度
    hull = cv2.convexHull(cnt)
    length = len(hull)
    return length

def maxAndSubMax(cnt):    #采用分治法计算最大和次大轮廓
    if len(cnt) == 1:
        return cnt[0], cnt[0]
    if len(cnt) == 2:
        if hull_length(cnt[0]) > hull_length(cnt[1]):
            return cnt[0], cnt[1]
        else:
            return cnt[1], cnt[0]

    x1L, x2L = maxAndSubMax(cnt[:len(cnt) // 2])
    x1R, x2R = maxAndSubMax(cnt[len(cnt) // 2:])

    if hull_length(x1L) > hull_length(x1R):
        if hull_length(x2L) > hull_length(x1R):
            return x1L, x2L
        else:
            return x1L, x1R
    else:
        if hull_length(x1L) > hull_length(x2R):
            return x1R, x1L
        else:
            return x1R, x2R


# 图像路径
# im = r'img/rice_roi.jpg'
im = r'rice/whitespot.jpg'
# cv2.imshow('im_raw',im)
raw_gray = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
cv2.imshow('row_gray',raw_gray)

# 检测大米轮廓，测量尺寸
im2 = detect_rice(im)
# detect_yellow_spot()
# detect_white_spot(im2)
#detect_fracture()


cv2.destroyAllWindows()