import cv2
def draw_rectangle(event,x,y,flags,param):
    # global ix, iy
    if event==cv2.EVENT_LBUTTONDOWN:
    #     ix, iy = x, y
    #     print("point1:=", x, y)
    # elif event==cv2.EVENT_LBUTTONUP:
    #     print("point2:=", x, y)
    #     print("width=",x-ix)
    #     print("height=", y - iy)
    #     cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.circle(img, (x,y), 50, (255, 0, 0), -1)

img = cv2.imread("/Users/jarvix/Downloads/box.jpeg")  #加载图片
cv2.namedWindow('image',0)
cv2.resizeWindow("image", 640, 480)
cv2.setMouseCallback('image', draw_rectangle)
while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
