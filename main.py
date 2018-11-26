# coding:utf-8
import cv2
import numpy as np
import copy
import math

def tracking(image, click_center):
    r = 100
    r2 = r * 2
    startx = click_center[0] - r
    starty = click_center[1] - r
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0

    width = r2
    height = r2
    if startx + width >= image.shape[1]:
        startx = image.shape[1] - width - 1
    if starty + height >= image.shape[0]:
        starty = image.shape[0] - height - 1

    roi = copy.deepcopy(image)
    roi = roi[starty:starty+height, startx:startx+width]
    roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 检测红色区域并二值化
    mask1 = cv2.inRange(roiHSV, (0, 70, 50), (10, 255, 255))
    mask2 = cv2.inRange(roiHSV, (170, 70, 50), (180, 255, 255))
    mask = mask1 | mask2

    # 找到连通区域
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    gravity_centers = []
    for i in range(len(contours)):
        if len(contours[i]) < 10:
            continue
        tmp = contours[i].squeeze(1)
        xsum = 0
        ysum = 0
        for j in range(len(tmp)):
            xsum += tmp[j][0]
            ysum += tmp[j][1]
    
        gpx = int(xsum/len(tmp))
        gpy = int(ysum/len(tmp))
        gravity_centers.append((gpx+startx, gpy+starty))

    ret = click_center
    dist = 10000000000
    distX = 10000000000
    distY = 10000000000
    for i in range(len(gravity_centers)):
        if distX > abs(click_center[0]-gravity_centers[i][0]) and distY > abs(click_center[1]-gravity_centers[i][1]):
            newdist = math.sqrt((click_center[0]-gravity_centers[i][0])**2+(click_center[1]-gravity_centers[i][1])**2)
            if dist > newdist:
                distX = abs(click_center[0]-gravity_centers[i][0])
                distY = abs(click_center[1]-gravity_centers[i][1])
                dist = newdist
                ret = gravity_centers[i]

    return ret

def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ret = tracking(image, (x,y))
        if ret:
            cv2.circle(image, ret, 20, (255, 0, 0), -1)

image = cv2.imread('/Users/jarvix/Downloads/box.jpeg')#'/home/jianan/Desktop/clip.jpeg')

cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 640, 480)
cv2.setMouseCallback('image', select_point)  # 设置回调函数

while True:
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
