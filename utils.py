import math
import numpy as np


def rotateByZ(x, y, thetaz):
    rz = thetaz * math.pi / 180
    outx = math.cos(rz) * x - math.sin(rz) * y
    outy = math.sin(rz) * x + math.cos(rz) * y
    return outx, outy

def rotateByY(x, z, thetay):
    ry = thetay * math.pi / 180
    outx = math.cos(ry) * x + math.sin(ry) * z
    outz = math.cos(ry) * z - math.sin(ry) * x
    return outx, outz

def rotateByX(y, z, thetax):
    rx = thetax * math.pi / 180
    outy = math.cos(rx) * y - math.sin(rx) * z
    outz = math.cos(rx) * z + math.sin(rx) * y
    return outy, outz


def rotateByVector(x,y,z,v,theta):
    r = theta * math.pi / 180
    c = math.cos(r)
    s = math.sin(r)
    vx, vy, vz = v
    new_x = (vx*vx*(1 - c) + c) * x + (vx*vy*(1 - c) - vz*s) * y + (vx*vz*(1 - c) + vy*s) * z
    new_y = (vy*vx*(1 - c) + vz*s) * x + (vy*vy*(1 - c) + c) * y + (vy*vz*(1 - c) - vx*s) * z
    new_z = (vx*vz*(1 - c) - vy*s) * x + (vy*vz*(1 - c) + vx*s) * y + (vz*vz*(1 - c) + c) * z
    return new_x,new_y,new_z


def get_plane(p1,p2,p3):
    # 三点确定一个平面
    x1,y1,z1 = p1[0],p1[1],p1[2]
    x2,y2,z2 = p2[0],p2[1],p2[2]
    x3,y3,z3 = p3[0],p3[1],p3[2]
    A = y1*(z2-z3)+y2*(z3-z1)+y3*(z1-z2)
    B = z1*(x2-x3)+z2*(x3-x1)+z3*(x1-x2)
    C = x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)
    D = -x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1)
    plane_paras = np.array([A, B, C, D])
    # 并得到平面的法向量
    v1 = p2-p1
    v2 = p3-p1
    norm_vec = np.cross(v1, v2)
    norm_vec = norm_vec/np.linalg.norm(norm_vec)
    return plane_paras, norm_vec

def angle_between_vectors(v1, v2):
    radian = np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    angle = radian*360/2/np.pi
    if angle > 90:
        angle = 180 - angle
    return angle
