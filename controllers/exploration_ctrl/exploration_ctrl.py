"""exploration_ctrl controller."""

from controller import Robot
import sys

sys.path.append("..")
from utils.movement import Movement

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get motors
lm = robot.getDevice("left wheel")
rm = robot.getDevice("right wheel")

move=Movement(lm, rm)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    pass

# Exit & cleanup code.
