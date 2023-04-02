'''
movement.py - movement functions for Pioneer 3-DX
'''

from enum import Enum
import numpy as np

class Movement():

    def __init__(self, leftMotor, rightMotor):

        self.lm=leftMotor
        self.rm=rightMotor
        
        # initialize motors
        self.lm.setPosition(float('inf'))
        self.rm.setPosition(float('inf'))

        # default motor velocity
        self.VEL=2

        # desired position accuracy
        self.EPSILON=0.1
        
        self.state=[]
        self.target=[0,0,0]

        # high level state - what is our goal right now?
        # stop: stopping
        # linear: moving to target in a straight line
        self.algostates=Enum("Algorithm",["stop","linear"])
        self.algostate=self.algostates.stop

        # low level state - what are we doing right now?
        # stop: stopping
        # the rest: moving forward/backward, turning left/right
        self.movestates=Enum("Movement",["stop","forward","backward","left","right"])
        self.movestate=self.movestates.stop

    def linmoveto(self, x, y, theta):
        '''
        move in a straight line toward a given x, y, theta state.
        
        x: robot x coordinate in meters
        y: robot y coordinate in meters
        theta: robot angle in radians.
        '''

        self.target=[x, y, theta]
        self.algostate=self.algostates.linear

    def update(self, poseEstimate):
        '''
        update states and motor signals
        '''
        # get new pose estimate
        self.state=poseEstimate
        # run algo update: this sets new algostate and new movestate.
        self.algoupdate()
        # run movement update: this sets new motor velocities.
        self.moveupdate()

    def algoupdate(self):
        if self.algostate==self.algostates.linear:
            self.linmoveupdate()
        else:
            # stop
            self.movestate=self.movestates.stop
        
    def moveupdate(self):
        '''
        update motor signals according to movestate
        '''
        if self.movestate==self.movestates.forward:
            self.lm.setVelocity(self.VEL)
            self.rm.setVelocity(self.VEL)
        elif self.movestate==self.movestates.backward:
            self.lm.setVelocity(-self.VEL)
            self.rm.setVelocity(-self.VEL)
        elif self.movestate==self.movestates.left:
            self.lm.setVelocity(self.VEL)
            self.rm.setVelocity(-self.VEL)
        elif self.movestate==self.movestates.right:
            self.lm.setVelocity(-self.VEL)
            self.rm.setVelocity(self.VEL)
        else:
            # stop
            self.lm.setVelocity(0)
            self.rm.setVelocity(0)

    def linmoveupdate(self):
        '''
        linear motion toward goal update
        '''
        
        if all(abs(self.state[:2]-self.target[:2])<self.EPSILON):
            # if x, y does match (on target), orient robot toward goal orientation.
            if abs(self.state[2]-self.target[2])<self.EPSILON:
                self.movestate=self.movestates.stop
                self.algostate=self.algostates.stop
            else:
                self.rotateto(self.target[2])
        else:
            # if x, y doesn't match (not on target), orient robot toward goal.
            # calculate target angle
            targetang=np.arctan2(self.target[1]-self.state[1], self.target[0]-self.state[0])
            if abs(self.state[2]-targetang)<self.EPSILON:
                # if on target angle, go forward
                self.movestate=self.movestates.forward
            else:
                # if not, orient robot
                self.rotateto(targetang)

    def rotateto(self, theta):
        '''
        rotate to given angle.

        theta: desired angle in radians.
        '''
        current=self.state[2]
        delta=theta-current
        if abs(delta)<self.EPSILON:
            self.movestate=self.movestates.stop
        else:
            # weird calculation to determine whether left or right is shorter
            if (1-(2*(delta//np.pi)))*np.sign(delta)+1:
                self.movestate=self.movestates.left
            else:
                self.movestate=self.movestates.right
