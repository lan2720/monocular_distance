import math

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



