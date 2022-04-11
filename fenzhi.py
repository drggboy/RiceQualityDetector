import cv2

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

    if x1L > x1R:
        if x2L > x1R:
            return x1L, x2L
        else:
            return x1L, x1R
    else:
        if x1L > x2R:
            return x1R, x1L
        else:
            return x1R, x2R

