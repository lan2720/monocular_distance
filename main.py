# coding:utf-8
import cv2
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from utils import rotateByZ, rotateByX, rotateByY
from show import plot_camera

marker_2d = []

def load_intrinsic_parameters(npz_file):
    with np.load(npz_file) as X:
        mtx, dist = [X[i] for i in ('mtx','dist')]
    return mtx, dist


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

    ## 检测红色区域并二值化
    #mask1 = cv2.inRange(roiHSV, (0, 70, 50), (10, 255, 255))
    #mask2 = cv2.inRange(roiHSV, (170, 70, 50), (180, 255, 255))
    #mask = mask1 | mask2
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    hsv = cv2.split(roiHSV)
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]
    ts = 0.5*255
    tv = 0.1*255
    th = 0*180/360
    thadd = 30*180/360

    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if s[i][j] < ts or v[i][j] < tv:
                continue
            if th + thadd > 180:
                if h[i][j] < th - thadd and h[i][j] > th + thadd - 180:
                    continue
            if th - thadd < 0:
                if h[i][j] < th - thadd + 180 and h[i][j] > th + thadd:
                    continue
            mask[i][j] = 255
    print("mask:", mask)

    # 找到连通区域
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return click_center

    cv2.drawContours(roi, contours, -1, (0,255,0), 3)
    cv2.imshow("roi", roi)
    
    # 找到各连通域的重心位置
    gravity_centers = []
    for i in range(len(contours)):
        if len(contours[i]) < 30:
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
    
    # 找到距离点击点最近的重心
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
                ret = tuple(gravity_centers[i])

    return ret

def select_point(event, x, y, flags, param):
    global marker_2d
    if event == cv2.EVENT_LBUTTONDOWN:
        marker_2d.append((x,y))


def get_camera_in_world(thetax, thetay, thetaz, p0):
    # p0 is t (which is extrinsic parameter of camera)
    x = copy.deepcopy(p0)
    x[0], x[1] = rotateByZ(x[0], x[1], -1 * thetaz)
    x[0], x[2] = rotateByY(x[0], x[2], -1 * thetay)
    x[1], x[2] = rotateByX(x[1], x[2], -1 * thetax)
    return -1*x
  
def draw(img, p0, imgpts):
    img = cv2.line(img, p0, tuple(imgpts[0].ravel()), (255,0,0), 5) # blue x axis
    img = cv2.line(img, p0, tuple(imgpts[1].ravel()), (0,255,0), 5) # green y 
    img = cv2.line(img, p0, tuple(imgpts[2].ravel()), (0,0,255), 5) # red z
    return img

cap = cv2.VideoCapture(0)
#image = cv2.imread('/home/jianan/Pictures/box.jpg')
cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 640, 480)
cv2.namedWindow("tmp", 0)
cv2.resizeWindow("tmp", 640, 480)
cv2.setMouseCallback('image', select_point)  # 设置回调函数

marker_3d = np.array([[0,0,0],[150,0,0],[0,200,0],[150,200,0]], dtype=np.float32).reshape(-1,1,3)
axis = np.float32([[30,0,0], [0,30,0], [0,0,30]]).reshape(-1,3)

mtx, dist = load_intrinsic_parameters('webcam_calibration_ouput.npz')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
while True:
    _, image = cap.read() 
    for i in range(len(marker_2d)):
        print('tracking: point %d =' % i, marker_2d[i])
        ret = tracking(image, marker_2d[i])
        marker_2d[i] = ret
        cv2.circle(image, ret, 4, (255,0,0), -1)
    
    if len(marker_2d) == 4:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(marker_3d, np.array(marker_2d, dtype=np.float32).reshape(-1,1,2), mtx, dist)
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        tmp = copy.deepcopy(image)
        tmp = draw(tmp, marker_2d[0], imgpts)
        cv2.imshow('tmp', tmp)
        rotM,_ = cv2.Rodrigues(rvecs)
        r11 = rotM[0][0]
        r12 = rotM[0][1]
        r13 = rotM[0][2]
        r21 = rotM[1][0]
        r22 = rotM[1][1]
        r23 = rotM[1][2]
        r31 = rotM[2][0]
        r32 = rotM[2][1]
        r33 = rotM[2][2]
        thetaz = math.atan2(r21, r11) / math.pi * 180
        thetay = math.atan2(-1 * r31, math.sqrt(r32*r32 + r33*r33)) / math.pi * 180
        thetax = math.atan2(r32, r33) / math.pi * 180
        p0 = tvecs.reshape(3,)
        cw = get_camera_in_world(thetax, thetay, thetaz, p0)
        print("camera pos in world axis:", cw)
        # Create plot
        plot_camera(ax, cw)
        
    cv2.imshow('image', image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
