"""exploration_ctrl controller."""

from controller import Robot
import numpy as np
import cv2
import sys

sys.path.append("..")
from utils.movement import Movement
from utils.posest import posest
from utils.planner import pathplan

TAG_METHOD = 'least_square'

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get motors
lm = robot.getDevice("left wheel")
rm = robot.getDevice("right wheel")
# get camera
camera = robot.getDevice("camera")
camera.enable(timestep)

front_cam = robot.getDevice("front camera")
front_cam.enable(timestep)

move = Movement(lm, rm)

# points=[[2,1,-1],
#         [0,0,1],
#         [1,1,0]]
#points=[[0,0]]
def check_aruco(img,pose,aruco_dict=cv2.aruco.DICT_4X4_100):
    cameraMatrix = np.array([[528, 0, 360],[0, 528, 360],[0, 0, 1]])
    distCoeffs = np.array([0, 0, 0, 0, 0])
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
    parameters =  cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if ids is not None:
        '''
        rvec, tvec ,_ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
        robotx,roboty,robottheta=pose
        marker_x = robot_x + tvec[0][0][0]*np.cos(robottheta) - tvec[0][0][2]*np.sin(robottheta)
        marker_y = robot_y + tvec[0][0][0]*np.sin(robottheta) + tvec[0][0][2]*np.cos(robottheta)
        m
        '''
        return True, ids[0][0]
    else:
        return False, None

i=0
MARKER_NUM = 4
seen_markers = []
#move.linmoveto(*points[0])

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    img = camera.getImage()
    img = np.frombuffer(img, np.uint8).reshape(camera.height, camera.width, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    front_img = front_cam.getImage()
    front_img = np.frombuffer(front_img, np.uint8).reshape(front_cam.height, front_cam.width, 4)
    front_img = cv2.cvtColor(front_img, cv2.COLOR_BGRA2RGB) 

    state = posest(img, TAG_METHOD) 

    flag, id = check_aruco(front_img,state)
    if flag:
        if id not in seen_markers:
            seen_markers.append(id)
        if len(seen_markers) == MARKER_NUM:
            print("All markers seen")

    if i==0:
        stats, path=pathplan(tuple(state[:2]))
        points=path
    
    move.update(state)
    
    print(f"Pose estimate: {np.array2string(state, precision=5, floatmode='fixed', suppress_small=True)}, \tTarget: {move.target}, \tState: {move.algostate} {move.movestate}")
    
    if move.algostate==move.algostates.stop:
        print(i)
        i+=1
        if i==len(points):
            break
        else:
            if points[i]=="waypoint":
                move.turnaround()
            else:
                move.linmoveto(*points[i])
        
    
    
    

# Exit & cleanup code.
