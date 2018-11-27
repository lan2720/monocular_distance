import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy
from utils import rotateByZ, rotateByX, rotateByY


def hehe(ax, camera_point):
    data = np.array([[0,0,0], # P1
                     [150,0,0], # P2
                     [0,200,0], # P3
                     [150,200,0]]) # P4

    ax.scatter(data[:,0], data[:,1], data[:,2], c="red")
    ax.scatter([camera_point[0]], [camera_point[1]], [camera_point[2]], c="blue")

    ax.set_zlim(0.0, 300.0)
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(0.0, 300.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect(1)
    plt.draw()
    plt.pause(0.1)
    plt.cla()

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
