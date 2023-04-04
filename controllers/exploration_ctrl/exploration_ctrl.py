"""exploration_ctrl controller."""

from controller import Robot
import numpy as np
import cv2
import sys

sys.path.append("..")
from utils.movement import Movement
from utils.posest import posest

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

move.linmoveto(2, 1, -1)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:

    img = camera.getImage()
    img = np.frombuffer(img, np.uint8).reshape(camera.height, camera.width, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    state = posest(img)
    move.update(state)
    
    print(f"Pose estimate: {state}, State: {move.algostate} {move.movestate}")
    
    

# Exit & cleanup code.
