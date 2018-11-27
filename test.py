# encoding: utf-8
 
'''
使用 matplotlib动态更新
'''
import time
 
from matplotlib import pyplot as plt
import numpy as np
 
plt.axis([0, 1000, 0, 1])
plt.ion()

# fig = plt.figure()#str(time.time()))
# fig.canvas.set_window_title('Window 3D')

try:
	while True:
	    x = []
	    y = []
	    for i in range(512):
	        x.append(i)
	        y.append(np.random.random())
	        # plt.pause(0.05)
	    plt.cla()
	    plt.plot(x, y)
	    plt.pause(0.03)
	    # plt.waitforbuttonpress(0) # this will wait for indefinite time
except KeyboardInterrupt:
	plt.close()