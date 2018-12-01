from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from main import get_plane

p1 = np.array([0,0,0])
p2 = np.array([1,0,1])
p3 = np.array([0,1,0])
d = get_plane(p1,p2,p3)
print("plane:", d)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
X=np.arange(0, 6, 1, dtype=np.int)
Y=np.arange(0, 6, 1, dtype=np.int)
X, Y = np.meshgrid(X, Y)
Z = -1*(d[0]*X+d[1]*Y+d[3])/d[2]
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        color='gray')
ax.set_zlim3d(0, 5)
plt.show()
