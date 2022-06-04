#整体显示
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
#读取第一张图像
# im_path = r'self_img/camera/1.jpg'
im_path = r'img/im_open.jpg'
img = cv2.imread(im_path, cv2.IMREAD_COLOR)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_h = img_hsv[..., 0]
cv2.imwrite('.\img\img_h.jpg',img_h)
# print(np.shape(img_h))
# cv2.imshow('img_h',img_h)
cv2.waitKey(0)
#获取图像尺寸
h,w=img.shape[0:2]
gray1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#BGR转换为灰度显示格式   #库函数，Y亮度
#自定义空白单通道图像，用于存放灰度图
gray2=np.zeros((h,w),dtype=img.dtype)#最大值
gray3=np.zeros((h,w),dtype=img.dtype)#平均值
gray4=np.zeros((h,w),dtype=img.dtype)#Gamma校正灰度处理
gray_B=np.zeros((h,w),dtype=img.dtype)#分量法B通道
gray_G=np.zeros((h,w),dtype=img.dtype)#分量法G通道
gray_R=np.zeros((h,w),dtype=img.dtype)#分量法R通道
gray_weighted=np.zeros((h,w),dtype=img.dtype)#加权平均值
#对原图像进行遍历，然后分别对B\G\R按比例灰度化
for i in range(h):
    for j in range(w):
        gray2[i,j]=max(img[i,j,0],img[i,j,1],img[i,j,2]) #最大值
for i in range(h):
    for j in range(w):
        gray3[i,j]=(int(img[i,j,0])+int(img[i,j,1])+int(img[i,j,2]))/3 #平均值
for i in range(h):
    for j in range(w):
        a=img[i,j,2]**(2.2)+1.5*img[i,j,1]**(2.2)+0.6*img[i,j,0]**(2.2) #分子
        b=1+1.5**(2.2)+0.6**(2.2) #分母
        gray4[i,j]=pow(a/b,1.0/2.2)  #开2.2次方根  #Gamma校正灰度处理

for i in range(h):
    for j in range(w):
        gray_B[i,j]=img[i,j,0] #分量法B通道

for i in range(h):
    for j in range(w):
        gray_G[i,j]=img[i,j,1] #分量法G通道

for i in range(h):
    for j in range(w):
        gray_R[i,j]=img[i,j,2] #分量法R通道

for i in range(h):
    for j in range(w):
        gray_weighted[i,j]=0.3 * img[i, j, 2] + 0.11 * img[i, j, 0] + 0.59 * img[i, j, 1] #加权平均值  0.3 R+0.11 G+0.59 B
#BGR转换为RGB显示格式，方便通过matplotlib进行图像显示
gray1= cv2.cvtColor(gray1,cv2.COLOR_BGR2RGB)
gray2= cv2.cvtColor(gray2,cv2.COLOR_BGR2RGB)
gray3= cv2.cvtColor(gray3,cv2.COLOR_BGR2RGB)
gray4= cv2.cvtColor(gray4,cv2.COLOR_BGR2RGB)
gray_B= cv2.cvtColor(gray_B,cv2.COLOR_BGR2RGB)
gray_G= cv2.cvtColor(gray_G,cv2.COLOR_BGR2RGB)
gray_R= cv2.cvtColor(gray_R,cv2.COLOR_BGR2RGB)
gray_weighted= cv2.cvtColor(gray_weighted,cv2.COLOR_BGR2RGB)
#显示图像
titles = ['cv2.cvtColor()', '最大值灰度化','平均值灰度化','Gamma校正灰度化',\
          '分量法R通道','分量法G通道','分量法B通道','加权平均值','img_h']  #标题
images = [gray1, gray2,gray3,gray4,gray_R,gray_G,gray_B,gray_weighted,img_h]   #图像对比显示
for i in range(8):
    plt.subplot(3,3,i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')#关闭坐标轴  设置为on则表示开启坐标轴
plt.subplot(3,3,9), plt.imshow(img_h, cmap='gray')
plt.title('HSV中H分量')
plt.axis('off')
plt.show()#显示图像
