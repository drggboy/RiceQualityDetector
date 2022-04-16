import cv2
import numpy as np

im = cv2.imread(r'img/rice_roi.jpg')
# cv2.imshow('image',im)
# cv2.waitKey(0)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('img/rice_roi_thresh.jpg',thresh)
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)

thresh_by = np.zeros((thresh.shape),dtype=np.uint8)
roi = thresh[327:469,153:346]
# cv2.imshow('roi',roi)
# cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # kernel大小，准备进行开运算
# thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算
roi_close = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)  # 闭运算

cv2.imshow('roi_close',roi_close)
# cv2.imshow('thresh_open',thresh_open)
# cv2.waitKey(0)

roi_close_open = cv2.morphologyEx(roi_close, cv2.MORPH_OPEN, kernel,3)
cv2.imshow('roi_close_open',roi_close_open)
cv2.waitKey(0)