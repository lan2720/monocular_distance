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

PERSONID = 0

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
    #print("box:", box)
    #cv2.drawContours(image, [box], 0, (0,0,255), 2)
    #print("area:", area)
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
        #print("trunk:", trunk)
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
        #print("left_arm/left_forearm:", left_arm/left_forearm)
        d.append(check_in_range(left_arm/left_forearm, minv, maxv))
    if right_arm and right_forearm:
        #print("right_arm/right_forearm:", right_arm/right_forearm)
        d.append(check_in_range(right_arm/right_forearm, minv, maxv))
    if left_thigh and left_shank:
        #print("left_thigh/left_shank:", left_thigh/left_shank)
        d.append(check_in_range(left_thigh/left_shank, minv, maxv))
    if right_thigh and right_shank:
        #print("right_thigh/right_shank:", right_thigh/right_shank)
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
        #print("trunk/ave_arm:", trunk/ave_arm)
        betcheck.append(check_in_range(trunk/ave_arm, 1.0, 5.0))
    if ave_leg:
        #print("trunk/ave_leg:", trunk/ave_leg)
        betcheck.append(check_in_range(trunk/ave_leg, 0.7, 5.0))
    if ave_arm and ave_leg:
        #print("ave_arm/ave_leg:", ave_arm/ave_leg)
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
    @param points: [25, 3]
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


def iou(bbox1, bbox2):
    """
    @param bbox: (x, y, w, h)
    """
    x1min, y1min, w1, h1 = bbox1
    x1max, y1max = x1min+w1-1, y1min+h1-1
    x2min, y2min, w2, h2 = bbox2
    x2max, y2max = x2min+w2-1, y2min+h2-1
    inter = max(0, (min(x1max, x2max)-max(x1min, x2min)+1))*max(0, (min(y1max, y2max)-max(y1min, y2min)+1))
    union = w1*h1+w2*h2-inter
    iou = inter/union*1.0
    return iou
    

def filter_keypoint_by_zero(keypoint):
    """
    @param keypoint: [25, 3]
    @return new_keypoint: [n, 2] where n represents non-zero row
    """
    new_keypoint = [k for k in keypoint]
    new_keypoint = list(filter(lambda i: i[2] != 0, new_keypoint))
    return np.array(new_keypoint)[:,:2].astype(np.int)


def create_new_tracker(keypoint, image, trackers, bboxes):
    global PERSONID
    new_keypoint = filter_keypoint_by_zero(keypoint)
    bbox = cv2.boundingRect(new_keypoint)
    pose_tracker = cv2.TrackerKCF_create()
    pose_tracker.init(image, bbox)
    print("新骨架出现: personid=", PERSONID)
    trackers.setdefault(PERSONID, pose_tracker)
    bboxes.setdefault(PERSONID, bbox)
    PERSONID += 1


def init_trackers(keypoints, image):
    trackers, bboxes = {},{}
    for i in range(len(keypoints)):
        create_new_tracker(keypoints[i], image, trackers, bboxes)
    return trackers, bboxes


def draw_rect(bbox, personid, image):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
    cv2.putText(image, "PERSONID=%d"%personid, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)


def check_new_keypoint(keypoint, current_bboxes):
    # 将新keypoints的bbox同现有的bbox进行比较，判断该keypoint是否是新出现的
    new_keypoint = filter_keypoint_by_zero(keypoint)
    new_box = cv2.boundingRect(new_keypoint)
    area_track = []
    for box in current_bboxes:
        iouarea = iou(new_box, box)
        area_track.append(iouarea)
    top = sorted(area_track, reverse=True)[0]
    print("top score:", top)
    if top > 0.5:
        return False
    else:
        return True
    

def main():
    global PERSONID
    cap = cv2.VideoCapture(0)
    model = load_openpose_params()

    trackers = {}
    bboxes = {}
    frame_cnt = 0
    while True:
        _, image = cap.read()
        if image is None:
            print("Video finish")
            break
        frame_cnt += 1
        print("==============new frame===============")
        keypoints = get_keypoints(model, image) 
        for kp in keypoints:
            draw_keypoints(kp, image)
        if len(trackers) == 0:
            trackers, bboxes = init_trackers(keypoints, image)
        else:
            if frame_cnt % 5 != 0: # do tracking
                del_list = []
                for k in trackers.keys():
                    ok, bbox = trackers[k].update(image)
                    if not ok:
                        # 如果跟踪对象消失在视野中则删除
                        del_list.append(k)
                    else:
                        # 如果跟踪对象还处在视野中则更新跟踪框
                        bboxes[k] = bbox
                for k in del_list:
                    del trackers[k]
                    del bboxes[k] 
                    print("旧骨架消失: personid=", k)
            else: # do detect keypoint
                # 判断是否有新骨架出现
                for keypoint in keypoints:
                    # 1. 将新骨架的keypoints(n个)得到n个新bbox
                    # 2. 将这n个新bbox和目前现存的bboxes逐个计算IOU，如果IOU太小的就说明该骨架是新出现的
                    flag = check_new_keypoint(keypoint, bboxes.values())
                    if flag:
                        # 3. 将新出现的骨架加入到trackers和bboxes中
                        create_new_tracker(keypoint, image, trackers, bboxes)
                
        for pid, bbox in bboxes.items():
            draw_rect(bbox, pid, image)
        cv2.imshow("output", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
