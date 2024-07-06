import cv2 as cv
import glob
import numpy as np

chessBoardSize=(8,6)
frameSize=(640,480)

criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,0.001)

objp=np.zeros((chessBoardSize[0]*chessBoardSize[1],3),np.float32)
objp[:,:2]=np.mgrid[0:chessBoardSize[0],0:chessBoardSize[1]].T.reshape(-1,2)
size_of_chessboard_squares_mm=20
objp=objp*size_of_chessboard_squares_mm

objpoints=[]
imgpoints=[]

images=glob.glob('distorted.jpeg')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessBoardSize, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, chessBoardSize, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)
cv.destroyAllWindows()


# if len(imgpoints) > 0:
    # ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    # print(cameraMatrix)
# else:
#     print("No valid chessboard corners detected in any images.")
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print(dist)