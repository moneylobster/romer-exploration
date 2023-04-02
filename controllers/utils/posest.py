'''
posest.py - pose estimation functions
'''

import numpy as np
import cv2
import yaml
import ast

# Load ceiling params
with open('params.yaml') as f:
    loadedparams = yaml.safe_load(f)

GRID_SIZE = loadedparams.get('grid_size')

with open('../../misc/tagsettings.txt', 'r') as f:
    tag_set = f.read()
    tag_dict, tag_size = ast.literal_eval(tag_set)

# Load camera parameters
with open('../calib_supervisor/calibration.yaml') as f:
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
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    tag_errors = []
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
            for i,id in enumerate(ids):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], tag_size*0.01, matrix_coefficient, distortion_coefficient)
                tvec[0][0][1] = -tvec[0][0][1]
                position_est = tag_coords[id] + tvec[0][0][1::-1]
                
    return position_est
