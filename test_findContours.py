import cv2
import numpy as np
## 测试findContours
img = cv2.imread(r'img\test.png')
img = cv2.resize(img, (200,200))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',imgray)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('imageshow', img)  # 显示返回值image，其实与输入参数的thresh原图没啥区别
cv2.waitKey(0)

img = cv2.drawContours(img, contours, -1, (0, 255, 0), -1)  # img为三通道才能显示轮廓
cv2.imshow('123', img)
cv2.waitKey(0)
cv2.destroyAllWindows()