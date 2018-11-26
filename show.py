import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
 
# Create data
#N = 60
#g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N),0.4+0.1*np.random.rand(N))
#g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N),0.1*np.random.rand(N))
#g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N),0.3*np.random.rand(N))
data = np.array([[0,0,0], # P1
                 [0,200,0], # P2
                 [150,0,0], # P3
                 [150,200,0]]) # P4

color = "red" #, "green", "blue")
#groups = ("coffee", "tea", "water") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 

ax.scatter(data[:,1], data[:,0], data[:,2], c=color)
#ax.quiver(300,300,300,-50,-50,-50)
ax.set_zlim(0.0, 300.0)
ax.set_xlim(0.0, 300.0)
ax.set_ylim(300.0, 0.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect(1)

plt.title('realtime camera position and direction')
plt.show()
