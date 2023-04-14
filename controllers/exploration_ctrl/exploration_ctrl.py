"""exploration_ctrl controller."""

from controller import Robot
import numpy as np
import cv2
import sys

sys.path.append("..")
from utils.movement import Movement
from utils.posest import posest

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

move = Movement(lm, rm)

# points=[[2,1,-1],
#         [0,0,1],
#         [1,1,0]]
points=[[0,0,1]]
i=0

move.linmoveto(*points[0])

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    img = camera.getImage()
    img = np.frombuffer(img, np.uint8).reshape(camera.height, camera.width, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    state = posest(img, TAG_METHOD) 
    move.update(state)

    print(f"Pose estimate: {state}, State: {move.algostate} {move.movestate}")
    
    # TODO skips over two steps when movestate is stop, fix.
    if move.algostate==move.algostates.stop:
        move.linmoveto(*points[i])
        i+=1
        print(i)
        if i==len(points):
            move.turnaround()
        elif i==len(points)+1:
            break
    
    
    
    

# Exit & cleanup code.
