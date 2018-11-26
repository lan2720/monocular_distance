# coding:utf-8
import cv2
import numpy as np

image = cv2.imread('/home/jianan/Desktop/clip.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#COLOR_BGR2HLS)
hsv = cv2.split(image)
h = hsv[0]
s = hsv[1]
v = hsv[2]
print(h)
print(s)
print(v)

binary = np.zeros(image.shape[:2], dtype=np.uint8)

mask1 = cv2.inRange(image, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(image, (170, 70, 50), (180, 255, 255))
mask = mask1 | mask2
#ts = 0.5*255 # s阈值，小于该值不判断
#tv = 0.1*255 # v阈值，小于该值不判断
#th = 0*180/360
#thadd = 30*180/360
#
#for i in range(image.shape[0]):
#    for j in range(image.shape[1]):
#        #if s[i][j] < ts or v[i][j] < tv:
#        #    continue
#        if h[i][j] < 3 or h[i][j] > 175:
#            binary[i][j] = 255


while True:
    cv2.imshow('image', mask)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
