import cv2
import obj_detect
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    fracture_path = r'rice\fracture.jpg'
    # fracture_path = r'rice\yellow.jpg'
    fracture = cv2.imread(fracture_path, cv2.IMREAD_COLOR)

    # 开运算（去噪）后求roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # kernel大小，准备进行开运算
    fracture_open = cv2.morphologyEx(fracture, cv2.MORPH_OPEN, kernel)  # 开运算
    roi = obj_detect.detect_rice(fracture_open)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh_otsu, roi_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_and(fracture, fracture, mask=roi_mask)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # 断面检测
    detect_fracture(img)
    cv2.destroyAllWindows()
