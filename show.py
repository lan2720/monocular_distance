import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy
from utils import rotateByZ, rotateByX, rotateByY, get_plane

def plot_camera(ax, camera_point):
    data = np.array([[0,0,0], # P1
                     [150,0,0], # P2
                     [0,200,0], # P3
                     [150,200,0]]) # P4

    ax.scatter(data[:,0], data[:,1], data[:,2], c="red")
    ax.scatter([camera_point[0]], [camera_point[1]], [camera_point[2]], c="blue")

    ax.set_zlim(0.0, 600.0)
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(0.0, 600.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect(1)
    plt.draw()
    plt.pause(0.1)
    plt.cla()


def plot_arrow(ax, origin, vec):
    x, y, z = np.meshgrid(origin[0], origin[1], origin[2])
    u, v, w = np.split(vec.reshape(-1,1), vec.shape[0], axis=0)
    ax.quiver(x, y, z, u, v, w, length=50, normalize=True)



#def point_in_plane(pvec, p):
#    v = pvec[0]*p[0]+pvec[1]*p[1]+pvec[2]*p[2]+pvec[3]
#    if v == 0:
#        return True
#    else:
#        return False
#
#
#def plot_person_plane(ax, p1, p2, p3):
#    # points: should contain 3 points
#    m = get_plane(p1, p2, p3)
#    print("plane func:", m)
#    xmin = min(p1[0],p2[0],p3[0])
#    xmax = max(p1[0],p2[0],p3[0])
#    ymin = min(p1[1],p2[1],p3[1])
#    ymax = max(p1[1],p2[1],p3[1])
#    zmin = min(p1[2],p2[2],p3[2])
#    zmax = max(p1[2],p2[2],p3[2])
#    X = np.arange(xmin, xmax, 1, dtype=np.int)
#    Y = np.arange(ymin, ymax, 1, dtype=np.int)
#    X, Y = np.meshgrid(X, Y)
#    Z = -1*(m[0]*X+m[1]*Y+m[3])/m[2]
#    #Z = np.clip(Z, zmin, zmax)
#    X = X.reshape(-1)
#    Y = Y.reshape(-1)
#    Z = Z.reshape(-1)
#    tmpx, tmpy, tmpz = [], [], []
#    for x,y,z in zip(X,Y,Z):
#        if point_in_plane(m, [x,y,z]):
#            tmpx.append(x)
#            tmpy.append(y)
#            tmpz.append(z)
#    X = np.array(tmpx)[:121].reshape(11, 11)
#    Y = np.array(tmpy)[:121].reshape(11, 11)
#    Z = np.array(tmpz)[:121].reshape(11, 11)
#    print("X:", X)
#    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='gray')

def main():
    # Create data
    #N = 60
    #g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N),0.4+0.1*np.random.rand(N))
    #g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N),0.1*np.random.rand(N))
    #g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N),0.3*np.random.rand(N))
    data = np.array([[0,0,0], # P1
                     [0,150,0], # P2
                     [200,0,0], # P3
                     [200,150,0]]) # P4

    color = "red" #, "green", "blue")
     
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 

    ax.scatter(data[:,0], data[:,1], data[:,2], c=color)
    p  = [1,2,3]#[[100,0,0],[0,100,0],[0,0,100]])
    x = copy.deepcopy(p) #[None]*3
    #x[0], x[1], x[2] = rotateByVector(x[0],x[1],x[2],[0,0,1], 90)

    x[1], x[2] = rotateByX(x[1], x[2], 90)
    x[0], x[2] = rotateByY(x[0], x[2], -90)
    x[0], x[1] = rotateByZ(x[0], x[1], -90)
    print("new x:", x)

    ax.quiver(0,0,0,100,0,0)
    ax.quiver(0,0,0,0,100,0)
    ax.quiver(0,0,0,0,0,100)

    ax.quiver(0,0,0,p[0],p[1],p[2])
    ax.quiver(0,0,0,x[0],x[1],x[2])

    ax.set_zlim(0.0, 300.0)
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(300.0, 0.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect(1)

    plt.title('realtime camera position and direction')
    plt.show()

if __name__ == '__main__':
    main()
