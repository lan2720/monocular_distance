# coding:utf-8
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import math
from mpl_toolkits.mplot3d import axes3d
from utils import rotateByZ, rotateByX, rotateByY, get_plane, angle_between_vectors
from show import plot_camera, plot_arrow#, plot_person_plane
import sys
sys.path.append('/data1/Softwares/openpose/build/python/openpose');
import openpose as op


ground_2d = []


def load_openpose_params():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.5
    # If GPU version is built, and multiple GPUs are available, set the ID here
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["default_model_folder"] = "/data1/Softwares/openpose/models/"
    # Construct OpenPose object allocates GPU memory
    openpose = op.OpenPose(params)
    return openpose


def load_intrinsic_K(npz_file):
    with np.load(npz_file) as X:
        mtx = X['K']
    return mtx.astype(np.float)


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
    global ground_2d
    if event == cv2.EVENT_LBUTTONDOWN:
        ground_2d.append((x,y))


def get_camera_origin_in_world(thetax, thetay, thetaz, p0):
    # p0 is t (which is extrinsic parameter of camera)
    x = copy.deepcopy(p0)
    x[0], x[1] = rotateByZ(x[0], x[1], -1 * thetaz)
    x[0], x[2] = rotateByY(x[0], x[2], -1 * thetay)
    x[1], x[2] = rotateByX(x[1], x[2], -1 * thetax)
    return -1*x


def get_person_origin_in_world(thetax, thetay, thetaz, p0, cw):
    # p0: the person origin in camera coordinate
    # cw: the camera position in world coordinate
    # return: the person origin in world coordinate
    x = copy.deepcopy(p0)
    x[0], x[1] = rotateByZ(x[0], x[1], -thetaz)
    x[0], x[2] = rotateByY(x[0], x[2], -thetay)
    x[1], x[2] = rotateByX(x[1], x[2], -thetax)
    x += cw
    return x


def draw(img, p0, imgpts):
    cv2.line(img, p0, tuple(imgpts[0].ravel()), (255,0,0), 5) # blue x axis
    cv2.line(img, p0, tuple(imgpts[1].ravel()), (0,255,0), 5) # green y 
    cv2.line(img, p0, tuple(imgpts[2].ravel()), (0,0,255), 5) # red z


def openpose_keypoint(model, image):
    rs = 2
    ls = 5
    rw = 9
    lw = 12
    #ne = 1
    #mh = 8
    # shoulder, 2,5 # waist, 9,12  
    keypoints, output_image = model.forward(image, True)
    results = []
    for i in range(len(keypoints)):
        right_shoulder = keypoints[i][rs]
        left_shoulder = keypoints[i][ls]
        right_waist = keypoints[i][rw]
        left_waist = keypoints[i][lw]
        #midhip = keypoints[i][mh]
        if right_shoulder[2] == 0 or left_shoulder[2] == 0 or right_waist[2] == 0 or left_waist[2] == 0:
            results.append([])
        else:
            right_shoulder = tuple(right_shoulder[:2].astype(np.int).tolist())
            left_shoulder = tuple(left_shoulder[:2].astype(np.int).tolist())
            right_waist = tuple(right_waist[:2].astype(np.int).tolist())
            left_waist = tuple(left_waist[:2].astype(np.int).tolist())
            results.append([right_shoulder, right_waist, left_shoulder, left_waist])
    return results 


def main():
    cap = cv2.VideoCapture('20181205clip.avi')#'/data1/Project/Jail/给杨博士/法制行为录像/倒地.mp4')#'倒地自制_clip.avi')#'/data1/Project/Jail/给杨博士/法制行为录像/倒地.mp4')
    #image = cv2.imread('screenshot.jpg')

    # Load openpose to detect human keypoints
    model = load_openpose_params()

    # Make figures ready for illustration
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 640, 480)
    cv2.namedWindow("tmp", 0)
    cv2.resizeWindow("tmp", 640, 480)
    cv2.setMouseCallback('image', select_point)  # 设置回调函数
    
    # 地面区域，宽=7块瓷砖，高=3.6块瓷砖，每块瓷砖=40cm
    # 人体黄金比例: 上半身=0.382，取0.45*175=78，肩宽=36, 腰宽=30
    #marker_3d = np.array([[0,0,0],[140,0,0],[0,280,0],[140,280,0], [0,0,0], [78,3,0], [0,36,0], [78, 33, 0]], dtype=np.float32).reshape(-1,1,3)
    marker_3d = np.array([[0,0,0],[80,0,0],[0,80,0],[80,80,0], [0,0,0], [65,3,0], [0,36,0], [65, 33, 0]], dtype=np.float32).reshape(-1,1,3)
    axis = np.float32([[30,0,0], [0,30,0], [0,0,30]]).reshape(-1,3)
    #mtx, dist = load_intrinsic_parameters('webcam_calibration_ouput.npz')
    mtx = load_intrinsic_K('hearingroom_intrinsic_K.npz')
    dist = np.zeros((1,5), dtype=np.float)
    while True:
        _, image = cap.read()
        if image is None:
            break
        detect_results = openpose_keypoint(model, image)
        # these four points is ground plane
        for i in range(len(ground_2d)):#len(marker_2d)):
            #print('tracking: point %d =' % i, marker_2d[i])
            #ret = tracking(image, marker_2d[i])
            #marker_2d[i] = ret
            cv2.circle(image, ground_2d[i], 4, (255,0,0), -1)
        # these four points is human upper body plane
        for i in range(len(detect_results)):
            if not detect_results[i]:
                continue
            for j in range(4):
                #print("detect:", detect_results[i][j])
                cv2.circle(image, detect_results[i][j], 4, (255,0,0), -1)
        
        if len(ground_2d) == 4:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(marker_3d[:4], np.array(ground_2d, dtype=np.float32).reshape(-1,1,2), mtx, dist)
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            tmp = copy.deepcopy(image)
            draw(tmp, ground_2d[0], imgpts)
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
            pc = tvecs.reshape(3,)
            # 获取相机在世界坐标系中的坐标
            cw = get_camera_origin_in_world(thetax, thetay, thetaz, pc)
            #print("camera pos in world axis:", cw)
            
            upper_body_in_world = []
            for i in range(len(detect_results)):
                marker_2d = detect_results[i]
                if not marker_2d:
                    continue
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(marker_3d[4:], np.array(marker_2d, dtype=np.float32).reshape(-1,1,2), mtx, dist)
                p0c = tvecs.reshape(3,)
                # 获取人体原点在世界坐标系中的坐标
                p0w = get_person_origin_in_world(thetax, thetay, thetaz, p0c, cw)
                #print("person pos in world axis:", p0w)
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
                thetazp = math.atan2(r21, r11) / math.pi * 180
                thetayp = math.atan2(-1 * r31, math.sqrt(r32*r32 + r33*r33)) / math.pi * 180
                thetaxp = math.atan2(r32, r33) / math.pi * 180
                upper_body_in_world.append([])
                upper_body_in_world[i].append(p0w)
                # 在人体坐标系中的所有点(除去原点)先顺向旋转thetaxp，再顺向旋转thetax
                for p in marker_3d[5:]:
                    tmp = copy.deepcopy(p).reshape(3,)
                    #print("before rotation:", tmp)
                    tmp[1], tmp[2] = rotateByX(tmp[1], tmp[2], thetaxp)
                    tmp[0], tmp[2] = rotateByY(tmp[0], tmp[2], thetayp)
                    tmp[0], tmp[1] = rotateByZ(tmp[0], tmp[1], thetazp)
                    tmp[0], tmp[1] = rotateByZ(tmp[0], tmp[1], -thetaz)
                    tmp[0], tmp[2] = rotateByY(tmp[0], tmp[2], -thetay)
                    tmp[1], tmp[2] = rotateByX(tmp[1], tmp[2], -thetax)
                    #print("after rotation:", tmp)
                    piw = p0w + tmp
                    upper_body_in_world[i].append(piw)
            for i in range(len(upper_body_in_world)):
                for p in upper_body_in_world[i]:
                    ax.scatter([p[0]], [p[1]], [p[2]], c="red")
        
            # plot camera
            plot_camera(ax, cw, ground_points=marker_3d[:4].squeeze(1))
            for i in range(len(upper_body_in_world)):
                # plot normal vector
                _, human_norm_vec = get_plane(upper_body_in_world[i][0], 
                                              upper_body_in_world[i][1],
                                              upper_body_in_world[i][2])
                tmp = 90 - angle_between_vectors(human_norm_vec, np.array([0,0,1]))
                print("person %d angle=%d" % (i, tmp))
                plot_arrow(ax, upper_body_in_world[i][0], human_norm_vec)
                # plot person plane
                #plot_person_plane(ax, upper_body_in_world[i][0], upper_body_in_world[i][1], upper_body_in_world[i][2])
            
        cv2.imshow('image', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
