# coding:utf-8
"""
许多动作识别的根本在于骨架的跟踪
因此需要在相邻两帧之间做判断，判断后一帧的骨架和前一帧的对应关系

1. 判断畸形骨架

"""

import math
import cv2
import numpy as np
from main_jail import load_openpose_params, openpose_keypoint

def distance_2d(point1, point2):
    """
    @param point1/point2: array whose shape = (2,), or list whose length=2
    @return dist: the distance between the two points in 2D plane
    """
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def check_in_range(x, minv, maxv):
    if x >= minv and x <= maxv:
        return True
    else:
        return False

def check_tiny_pose(image, pose, area_threshold=15000):
    """
    @param pose: 2D array which represents a human pose, shape=[25, 3]
    #return: bool, if the pose is tiny
    """
    filter_points = []
    for i in range(pose.shape[0]):
        if pose[i][2] != 0:
            filter_points.append(pose[i][:2])
    rect = cv2.minAreaRect(np.array(filter_points).astype(np.int))
    area = rect[1][0]*rect[1][1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print("box:", box)
    #cv2.drawContours(image, [box], 0, (0,0,255), 2)
    print("area:", area)
    if area < area_threshold:
        return True
    else:
        return False


def check_pose_normal(pose):
    """
    @param pose: 2D array which represents a human pose, shape=[25, 3]
    @return: bool, if the pose is normal or abnormal, True represents normal
    """
    # 通过脖子(neck)和中髋(midhip)得到该姿态和真实人体的大致缩放比例
    if pose[1][2] != 0 and pose[8][2] != 0:
        trunk = distance_2d(pose[1][:2], pose[8][:2])
        print("trunk:", trunk)
    else:
        print("没有检测到躯干")
        return False
    # 得到上手臂，下手臂在图中的长度
    left_arm, left_forearm, right_arm, right_forearm = None, None, None, None
    if pose[2][2] != 0 and pose[3][2] != 0:
        left_arm = distance_2d(pose[2][:2], pose[3][:2])
    if pose[3][2] != 0 and pose[4][2] != 0:
        left_forearm = distance_2d(pose[3][:2], pose[4][:2])
    if pose[5][2] != 0 and pose[6][2] != 0:
        right_arm = distance_2d(pose[5][:2], pose[6][:2])
    if pose[6][2] != 0 and pose[7][2] != 0:
        right_forearm = distance_2d(pose[6][:2], pose[7][:2])
    # 得到大腿，小腿在图中的长度
    left_thigh, left_shank, right_thigh, right_shank = None, None, None, None
    if pose[9][2] != 0 and pose[10][2] != 0:
        left_thigh = distance_2d(pose[9][:2], pose[10][:2])
    if pose[10][2] != 0 and pose[11][2] != 0:
        left_shank = distance_2d(pose[10][:2], pose[11][:2])
    if pose[12][2] != 0 and pose[13][2] != 0:
        right_thigh = distance_2d(pose[12][:2], pose[13][:2])
    if pose[13][2] != 0 and pose[14][2] != 0:
        right_shank = distance_2d(pose[13][:2], pose[14][:2])
    # 四段上肢体和四段下肢的长度要基本相同
    minv, maxv = 0.5, 4.0
    d = []
    if left_arm and left_forearm:
        print("left_arm/left_forearm:", left_arm/left_forearm)
        d.append(check_in_range(left_arm/left_forearm, minv, maxv))
    if right_arm and right_forearm:
        print("right_arm/right_forearm:", right_arm/right_forearm)
        d.append(check_in_range(right_arm/right_forearm, minv, maxv))
    if left_thigh and left_shank:
        print("left_thigh/left_shank:", left_thigh/left_shank)
        d.append(check_in_range(left_thigh/left_shank, minv, maxv))
    if right_thigh and right_shank:
        print("right_thigh/right_shank:", right_thigh/right_shank)
        d.append(check_in_range(right_thigh/right_shank, minv, maxv))
    if False in d:
        print("四肢内部比例不对")
        return False
    # 最后检查躯干和各部位的比例
    arm = list(filter(lambda i: i is not None, [left_arm, left_forearm, right_arm, right_forearm]))
    if len(arm) > 0:
        ave_arm = sum(arm)/len(arm)
    else:
        ave_arm = None
    leg = list(filter(lambda i: i is not None, [left_arm, left_forearm, right_arm, right_forearm]))
    if len(arm) > 0:
        ave_leg = sum(leg)/len(leg)
    else:
        ave_leg = None
    betcheck = []
    if ave_arm:
        print("trunk/ave_arm:", trunk/ave_arm)
        betcheck.append(check_in_range(trunk/ave_arm, 1.0, 5.0))
    if ave_leg:
        print("trunk/ave_leg:", trunk/ave_leg)
        betcheck.append(check_in_range(trunk/ave_leg, 0.7, 5.0))
    if ave_arm and ave_leg:
        print("ave_arm/ave_leg:", ave_arm/ave_leg)
        betcheck.append(check_in_range(ave_arm/ave_leg, 0.6, 1.2))
    
    if False in betcheck:
        print("躯干与四肢之间比例不对")
        return False
    else:
        return True


def get_keypoints(model, image):
    """
    Use openpose original model to detect keypoints, and do postprocess to make it better

    @param model: openpose original model
    @param image:
    @return filtered: a list of array
    """
    filtered = []
    keypoints = model.forward(image)
    for i in range(len(keypoints)):
        if check_tiny_pose(image, keypoints[i]):
            continue
        flag = check_pose_normal(keypoints[i])
        if flag:
            filtered.append(keypoints[i])
    return filtered


def draw_keypoints(points, image):
    """
    @param points: [25, 2]
    @param image: the image to draw
    """
    def _trans(point):
        return tuple(map(lambda i: int(i), point[:2]))
    color = (255, 0, 0)
    linewidth = 5
    points = list(map(lambda i: tuple(i.tolist()), points))
    for i in range(1,15):
        if points[i][2] == 0:
            continue
        pos = _trans(points[i])
        cv2.circle(image, pos, linewidth, color, -1) 
    # draw line
    #if points[0][2] != 0 and points[1][2] != 0: 
    #    cv2.line(image, _trans(points[0]), _trans(points[1]), color, linewidth)
    if points[1][2] != 0 and points[2][2] != 0: 
        cv2.line(image, _trans(points[1]), _trans(points[2]), color, linewidth)
    if points[2][2] != 0 and points[3][2] != 0: 
        cv2.line(image, _trans(points[2]), _trans(points[3]), color, linewidth)
    if points[3][2] != 0 and points[4][2] != 0: 
        cv2.line(image, _trans(points[3]), _trans(points[4]), color, linewidth)
    if points[1][2] != 0 and points[5][2] != 0:
        cv2.line(image, _trans(points[1]), _trans(points[5]), color, linewidth)
    if points[5][2] != 0 and points[6][2] != 0:
        cv2.line(image, _trans(points[5]), _trans(points[6]), color, linewidth)
    if points[6][2] != 0 and points[7][2] != 0:
        cv2.line(image, _trans(points[6]), _trans(points[7]), color, linewidth)
    if points[1][2] != 0 and points[8][2] != 0:
        cv2.line(image, _trans(points[1]), _trans(points[8]), color, linewidth)
    if points[8][2] != 0 and points[9][2] != 0:
        cv2.line(image, _trans(points[8]), _trans(points[9]), color, linewidth)
    if points[8][2] != 0 and points[12][2] != 0:
        cv2.line(image, _trans(points[8]), _trans(points[12]), color, linewidth)
    if points[9][2] != 0 and points[10][2] != 0:
        cv2.line(image, _trans(points[9]), _trans(points[10]), color, linewidth)
    if points[10][2] != 0 and points[11][2] != 0:
        cv2.line(image, _trans(points[10]), _trans(points[11]), color, linewidth)
    if points[12][2] != 0 and points[13][2] != 0:
        cv2.line(image, _trans(points[12]), _trans(points[13]), color, linewidth)
    if points[13][2] != 0 and points[14][2] != 0:
        cv2.line(image, _trans(points[13]), _trans(points[14]), color, linewidth)


def polygon_iou(box1, box2, image):
    """
    @param box1/box2: a tiled rectangle
    @param image: shape=[height, width, 3]
    @return iou
    """
    im1 = np.zeros(image.shape[:2], dtype = "uint8")
    im2 =np.zeros(image.shape[:2], dtype = "uint8")
    box1_mask = cv2.fillPoly(im1, box1, 255)
    box2_mask = cv2.fillPoly(im2, box2, 255)
    masked_and = cv2.bitwise_and(box1_mask, box2_mask, mask=im1)
    masked_or = cv2.bitwise_or(box1_mask, box2_mask)
     
    or_area = np.sum(np.float32(np.greater(masked_or,0)))
    and_area = np.sum(np.float32(np.greater(masked_and,0)))
    iou = and_area/or_area
    return iou


def filter_keypoint_by_zero(keypoint):
    """
    @param keypoint: [25, 3]
    @return new_keypoint: [n, 2] where n represents non-zero row
    """
    new_keypoint = [k for k in keypoint]
    new_keypoint = list(filter(lambda i: i[2] != 0, new_keypoint))
    return np.array(new_keypoint)[:,:2].astype(np.int)


def main():
    cap = cv2.VideoCapture(0)
    model = load_openpose_params()
    pose_tracker = cv2.TrackerKCF_create()

    # Read first frame.
    ok, image = cap.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
 
    keypoints = get_keypoints(model, image)
    
    # Initialize tracker with first frame and bounding box
    for i in range(len(keypoints)):
        new_keypoint = filter_keypoint_by_zero(keypoints[i])
        x,y,w,h = cv.boundingRect(new_keypoint)
        ok = pose_tracker.init(image, bbox)
 
    while True:
        _, image = cap.read()
        if image is None:
            break
        #if first_frame:
        #    ok = pose_tracker.init(image, bbox)
        #ok, bbox = tracker.update(image)
        print("==============new frame===============")
        keypoints = get_keypoints(model, image) 
        print("len:", len(keypoints))
        for i in range(len(keypoints)):
            draw_keypoints(keypoints[i], image)
        cv2.imshow("output", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
