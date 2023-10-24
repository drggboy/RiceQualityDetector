import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import grayscale as gy

'''
使用findContours方法查找对象轮廓
'''
def detect_objects(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # gray = gy.H_grayscale(im)
    # gray = gy.Hist_grayscale(im)  # 灰度化
    # gray = gy.eH_grayscale(im)
    # gray = gy.max_grayscale(im)    #最大值灰度化
    # gray = gy.clahe_grayscale(im)
    # gray = gy.eH_grayscale(gray)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (63, 63))  #kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  #kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # kernel大小，准备进行开运算
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,1)    #开运算1次

    # gray = cv2.GaussianBlur(gray,(5,5),0) #通过高斯滤镜过滤高频噪音
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #二值化
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    # 对二值化图片降噪
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
    # thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2)  # 开运算
    # cv2.imshow('thresh_open', thresh_open)

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
    # k = 0 ####
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

'''
通过梯度方式检测米粒内部断层
本例采用Scharr算子
关于Sobel和Scharr算子，请参阅：https://www.cnblogs.com/yibeimingyue/p/10878514.html
'''
def nothing(parm):
    pass

def detect_fracture(img):
    img_raw = img.copy()
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])     #由于使用plt.show(),所以需要使用rgb进行显示
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

    cv2.namedWindow("images",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("s1", "images", 57, 255, nothing)
    cv2.createTrackbar("s2", "images", 32, 255, nothing)
    while (1):
        # cv2.namedWindow('images')
        s1 = cv2.getTrackbarPos("s1", "images")
        s2 = cv2.getTrackbarPos("s2", "images")
        out_img = cv2.Canny(img_raw, s1, s2)
        cv2.imshow("images", out_img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
    cv2.destroyAllWindows()



# 图像路径
# im_path = r'img/img_h.jpg'
# im_path = r'img/im_open.jpg'
im_path = r'rice/whitespot.jpg'
# im_path = r'rice/yellow.jpg'--
# im_path = r'self_img/camera/9.jpg'
# im_path = r'self_img/camera/9_open.jpg'
# im_path = r'self_img/camera/3.jpg'
# im_path = r'self_img/phone/4.jpg'
im = cv2.imread(im_path, cv2.IMREAD_COLOR)

# 缩放图片
# percent = 500/im.shape[1]
# im = cv2.resize(im, None, fx=percent, fy=percent, interpolation=cv2.INTER_AREA)
# # im = cv2.resize(im, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
# cv2.imshow('im_raw',im)
# cv2.waitKey(0)

# 灰度图
# raw_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('row_gray',raw_gray)
# cv2.waitKey(0)



# # 检测大米轮廓，测量尺寸
# im2 = detect_rice(im)
# # 白点检测
# detect_white_spot(im2)
# # 黄点检测
# detect_yellow_spot(im2)

# 断面检测图像路径
# im_path = r'rice\fracture.jpg'
# im_path = r'img\fracture_open.jpg'
# im = cv2.imread(im_path,cv2.IMREAD_COLOR)
# cv2.imshow('im',im)
# cv2.waitKey(0)

# fracture_path = r'rice\fracture.jpg'

fracture_path = r'rice\yellow.jpg'
fracture = cv2.imread(fracture_path,cv2.IMREAD_COLOR)
# cv2.imshow('fracture',fracture)
# cv2.waitKey(0)

# 开运算（去噪）后求roi
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  #kernel大小，准备进行开运算
fracture_open = cv2.morphologyEx(fracture, cv2.MORPH_OPEN, kernel)    #开运算
roi = detect_rice(fracture_open)


roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
thresh_otsu, roi_mask = cv2.threshold(roi_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img = cv2.bitwise_and(fracture,fracture,mask=roi_mask)
cv2.imshow('img',img)
cv2.waitKey(0)

# 断面检测
detect_fracture(img)

cv2.destroyAllWindows()