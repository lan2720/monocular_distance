import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard Dimensions
cbrow = 6
cbcol = 9
objp = np.zeros((cbrow*cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbrow,0:cbcol].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('images_calib_1122/*.jpg') #['images_calib_1122/20181122_15h00m30s221577.jpg']#glob.glob('images_calib_1123/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbrow,cbcol),None)
    print(corners.shape)
    print(corners[:5])

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        print('obj:', objp.shape)
        print('corners type:', type(corners), 'shape:', corners.shape)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print('corners2:', corners2.shape)
        imgpoints.append(corners2)


        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (cbrow,cbcol), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(3000)

cv2.destroyAllWindows()

# Save parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("webcam_calibration_ouput.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


