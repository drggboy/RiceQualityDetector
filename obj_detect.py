import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_objects(im):
    # 对彩色图预处理
    # 开运算（去噪）后求roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  #kernel大小，准备进行开运算
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)    #开运算

    # 不同的灰度化方法
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # gray = gy.H_grayscale(im)
    # gray = gy.Hist_grayscale(im)  # 灰度化
    # gray = gy.eH_grayscale(im)
    # gray = gy.max_grayscale(im)    #最大值灰度化
    # gray = gy.clahe_grayscale(im)
    # gray = gy.eH_grayscale(gray)

    # 去噪
    # 对灰度图，使用开运算去噪，不同的核设置
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (63, 63))  #kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  #kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,1)    #开运算1次

    # 高斯过滤，去噪
    # gray = cv2.GaussianBlur(gray,(5,5),0) #通过高斯滤镜过滤高频噪音

    cv2.imshow('gray',gray)
    cv2.waitKey(0)

    # 二值化，自适应二值化
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #二值化
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)

    # 对二值化图片降噪
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
    # thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2)  # 开运算
    # cv2.imshow('thresh_open', thresh_open)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 查找轮廓
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算最长、次长轮廓
    # cnt1,cnt2=maxAndSubMax(contours)
    # print('max=',hull_length(cnt1))
    # print('submax=',hull_length(cnt2))

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
    # k = 0
    for i in range(len(contours)):
        cur_index = len(contours) -1 - i # 从最外层的轮廓开始检索，suppose轮廓排序是先里后外
        # cur_index = i
        cnt = contours[cur_index]
        hull = cv2.convexHull(cnt) #查找轮廓的凸包多边形
        hull_area = cv2.contourArea(hull)
        img_area = im.shape[0]*im.shape[1]
        length = len(hull)
        # 过滤凸包点数小于20的轮廓，进一步去除噪音
        if hull_area > img_area/5 and length >20:
        # if length > 20:
        # if cur_index == 3:
        #     print(cur_index)
            if cur_index in painted:
                continue
            # k = k+1   ####
            # print(k)  #####
            # print('max_index=',cur_index)
            # cv2.drawContours(im, [contours[cur_index]], -1, (0, 0, 255),-1)
            # cv2.imshow('3',im)
            painted.append(hierarchy[0][cur_index][3]) #把当前检索到的轮廓的内嵌轮廓放入已绘制列表，避免重复绘制
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
    # print('满足条件的轮廓：',k)
    # print('len(obj)',len(objects))
    return objects

#检测大米轮廓，测量尺寸
def detect_rice(im):     #返回感兴趣目标(彩色的)
    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # kernel大小，准备进行开运算
    im1 = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)  # 开运算
    # cv2.imshow('im_open',im)
    # cv2.waitKey(0)
    # cv2.imwrite("im_open.jpg",im)

    im2 = im1.copy()

    objs = detect_objects(im)
    for obj in objs:
        #绘制目标最小外接矩形
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
        # 提取出感兴趣区域,黑白色
        im_hull_roi = np.zeros((im.shape), dtype=np.uint8)
        cv2.fillPoly(im_hull_roi, [obj.get('hull')], (255, 255, 255))
        # cv2.imshow('im_hull_roi',im_hull_roi)

        # im_cnt_roi = np.zeros((im.shape), dtype=np.uint8)
        # cv2.fillPoly(im_cnt_roi, [obj.get('cnt')], (255, 255, 255))
        # cv2.imshow('im_cnt_roi',im_cnt_roi)
        # median = cv2.GaussianBlur(im_hull_roi, (5, 5), 1)}}}}
        # cv2.imshow('median',median)

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

    cv2.imshow('blue_mark',im)   #输入的目标图像
    cv2.waitKey(0)
    cv2.imshow('rice_roi_color', im2)    #只显示出感兴趣区域,其他为黑色
    cv2.waitKey(0)
    # roi_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    # thresh_otsu, roi_mask = cv2.threshold(roi_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('rice_roi', im_hull_roi)  # 只显示出感兴趣区域,其他为黑色
    cv2.waitKey(0)
    return im2

if __name__ == "__main__":
    img_path = r'rice\yellow.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    roi = detect_rice(img)