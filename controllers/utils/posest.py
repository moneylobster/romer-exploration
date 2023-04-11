'''
posest.py - pose estimation functions
'''

import numpy as np
import cv2
import yaml
import ast

# Load ceiling params
with open('../../misc/tagsettings.txt', 'r') as f:
    tag_set = f.read()
    tag_dict, tag_size = ast.literal_eval(tag_set)

# Load camera parameters
with open('../../misc/calibration.yaml') as f:
    loadeddict = yaml.safe_load(f)

matrix_coefficient = np.array(loadeddict.get('camera_matrix'))
distortion_coefficient = np.array(loadeddict.get('dist_coeff'))
camera_inv = np.linalg.inv(matrix_coefficient)

# Load tag coordinates
with open('../../misc/tagcoords.txt', 'r') as f:
    tag_coords = f.readlines()
tag_coords = tag_coords[0].split(",")
tag_coords = [i.replace("[","") for i in tag_coords]
tag_coords = [i.replace("]","") for i in tag_coords]
tag_coords = np.array(tag_coords, dtype=np.float32)
tag_coords = np.reshape(tag_coords, (-1,2))

# Define arucomarker dict
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL}
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_dict])

def posest(img):
    '''
    estimate robot pose based on image.

    img: RGB image as numpy array.
    '''
    
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    if np.all(ids is not None):
        if len(ids) > 2:
            b = np.array([[0]])
            A = np.zeros((1,4))
            for i in range(len(corners)):
                xm = tag_coords[ids[i],0][0]
                ym = tag_coords[ids[i],1][0]
                A = np.append(A,np.array([[-1,0,xm,ym],[0,1,-ym,xm]]),axis=0)
                # tried using inverse of transform, didn't work.
                #A = np.append(A,np.array([[-1,0,xm,ym],[0,-1,ym,-xm]]),axis=0)
                x,y = np.mean(corners[i][0], axis=0)
                v = np.matmul(camera_inv,np.array([[y],[x],[1]]))
                b = np.append(b,v[:2,:],axis=0)
            b = b[1:,:]
            A = A[1:,:]
            # x y cos(theta) sin(theta)
            res = np.linalg.lstsq(A,3.045*b,rcond=None)
            # normalize sin(theta) and cos(theta) to unit magnitude to fix any inconsistencies between the two variables.
            costheta=np.sign(res[0][2])*np.sqrt(res[0][2][0]**2/(res[0][2][0]**2+res[0][3][0]**2))
            sintheta=np.sign(res[0][3])*np.sqrt(res[0][3][0]**2/(res[0][2][0]**2+res[0][3][0]**2))
            # rotate x and y by theta for some reason?????
            position_est = np.array([[res[0][0][0]*costheta[0]-res[0][1][0]*sintheta[0], res[0][0][0]*sintheta[0]+res[0][1][0]*costheta[0], np.arctan2(sintheta[0], costheta[0])]])
        else:
            # find the first corner of the first tag
            x1,y1 = corners[0][0][0]
            # find the second corner of the first tag
            x2,y2 = corners[0][0][1]
            # find the angle between the two corners
            theta = np.arctan2(y2-y1,x2-x1)
            # find the center of the first tag
            x,y = np.mean(corners[0][0], axis=0)
            # find the position of the robot in the camera frame
            u,v  = tag_coords[ids[0],0][0], tag_coords[ids[0],1][0]
            # find the position of the robot in the world frame
            x_est = -(u-matrix_coefficient[0,2])*2.95+matrix_coefficient[0,0]*(x*cos(theta)+y*sin(theta))
            y_est = -(v-matrix_coefficient[1,2])*2.95+matrix_coefficient[1,1]*(-x*sin(theta)+y*cos(theta))
            position_est = np.array([[x_est,y_est,theta]])                
            # TODO implement this one
                
            # TODO improve this one, currently it misses angle estimate.
            # either get it from rvec or get it from corners urself.
            
                
    return position_est[0]
