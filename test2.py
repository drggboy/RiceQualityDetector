import numpy as np
import cv2 as cv


# canny边缘检测
cv.namedWindow("images")
def nothing():
    pass

cv.createTrackbar("s1","images",201,255,nothing)
cv.createTrackbar("s2","images",97,255,nothing)
img = cv.imread("rice/rice.jpg",0)
img = cv.resize(img,(800,650))

# cv.createTrackbar("s1","images",56,255,nothing)
# cv.createTrackbar("s2","images",19,255,nothing)
# img = cv.imread("rice/yellow.jpg",0)

while(1):
    s1 = cv.getTrackbarPos("s1","images")
    s2 = cv.getTrackbarPos("s2","images")
    out_img = cv.Canny(img,s1,s2)
    cv.imshow("images",out_img)
    k = cv.waitKey(1)
    if k==ord("q"):
        break
cv.destroyAllWindows()