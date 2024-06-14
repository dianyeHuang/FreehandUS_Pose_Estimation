'''
Author: xxx
Date: 2024-06-10 13:48:31
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-06-14 20:43:54
FilePath: /freehand_camera/cam_inercali.py
Description: 

    Intrisic paramerters calibration:
    refer to: https://blog.csdn.net/sinat_16643223/article/details/110442795
    
    Calibration results:
    - Left camera
    mtx:
    [[358.06222443,   0.,         309.14361973],
    [  0.,         357.74953614,  186.11295799],
    [  0.,           0.,           1.        ]]
    dist:
    [[-7.97898732e-02,  2.49530609e-01,  1.02409919e-03, -5.47208067e-05, -2.44939489e-01]]
    
    - Right camera
    mtx:
    [[358.63432955   0.         331.22704577]
    [  0.         358.63717784 186.44587027]
    [  0.           0.           1.        ]]
    dist:
    [[-0.08539184  0.28856155  0.00045643  0.0005938  -0.30752491]]
    
'''
import cv2
import numpy as np
import glob

# need to collect some images 
images = glob.glob("right_images/*.jpg") 


# Detect the corner point
objp = np.zeros((4 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)  # build a world coordinate 
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


obj_points = []  
img_points = []  

i=0;
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (640, 360)) # because in the real case, we set the image resolution to be 640 x 360
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 4), None)

    if ret:

        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  

        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (6, 4), corners, ret) 
        i+=1;
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# start calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx)      # intrinsic matrix
print("dist:\n", dist)    # distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # extrinsic rotation
print("tvecs:\n", tvecs ) # extrinsic translation

print("-----------------------------------------------------")
img = cv2.imread(images[2])
img = cv2.resize(img, (640, 360)) # xxx resize
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

print("------------------apply undistort function-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult3.jpg', dst1)


