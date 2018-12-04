import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('倒地自制.avi', fourcc, 20, (640, 480))

while True:
    # get a frame
    ret, frame = cap.read()
    # save a frame
    out.write(frame)
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
