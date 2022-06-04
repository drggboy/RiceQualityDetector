import cv2
import numpy as np
import matplotlib.pyplot as plt


im_path = r'self_img/camera/9.jpg'
im = cv2.imread(im_path)
percent = 500/im.shape[1]
im = cv2.resize(im, None, fx=percent, fy=percent, interpolation=cv2.INTER_AREA)

im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow("im_gray",im_gray)
cv2.waitKey(0)

# 使用自适应直方图均衡化
# 第一步：实例化自适应直方图均衡化函数
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))

# 第二步：进行自适应直方图均衡化
clahe = clahe.apply(im_gray)
ret = cv2.equalizeHist(im_gray)
# 第三步：进行图像的展示
cv2.imshow('clahe', np.hstack((im_gray, ret, clahe)))
cv2.waitKey(0)
# cv2.destroyAllWindows()



hist_first = cv2.calcHist([im_gray], [0], None, [256], [0, 256])
plt.subplot(1,2,1),plt.plot(hist_first)
plt.title('hist_first')
# plt.axis('off')#关闭坐标轴  设置为on则表示开启坐标轴

Imax = np.max(im_gray)
Imin = np.min(im_gray)
MAX = 255
MIN = 0
im_cs = (im_gray - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
cv2.imshow("im_cs", im_cs.astype("uint8"))
cv2.waitKey(0)

hist_imcs = cv2.calcHist(np.uint8(im_cs), [0], None, [256], [0, 256])
plt.subplot(1,2,2),plt.plot(hist_imcs)
plt.title('hist_imcs')
# plt.axis('off')#关闭坐标轴  设置为on则表示开启坐标轴
plt.show()