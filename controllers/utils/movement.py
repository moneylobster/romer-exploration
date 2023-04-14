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
        self.VEL=8
        # multiplier to slow robot down as it gets closer to target
        self.veldamper=1.0
        # desired position accuracy
        self.EPSILON=0.1
        # desired angle accuracy
        self.ANGEPSILON=0.1

        self.VELCUTOFF=10
        self.ANGCUTOFF=20
        
        self.state=[]
        self.target=[0,0,0]
        
        # high level state - what is our goal right now?
        # stop: stopping
        # linear: moving to target in a straight line
        self.algostates=Enum("Algorithm",["stop","linear","turnaround"])
        self.algostate=self.algostates.stop

        # low level state - what are we doing right now?
        # stop: stopping
        # the rest: moving forward/backward, turning left/right
        self.movestates=Enum("Movement",["stop","forward","backward","left","right"])
        self.movestate=self.movestates.stop

        # to track which step of turning around we are in.
        self.turnstate=0

    def linmoveto(self, x, y, theta=None):
        '''
        move in a straight line toward a given x, y, theta state.
        
        x: robot x coordinate in meters
        y: robot y coordinate in meters
        theta: robot angle in radians. If None, final pose does not matter.
        '''

        self.target=[x, y, theta]
        self.algostate=self.algostates.linear

    def turnaround(self):
        '''
        do a 360 degree rotation.
        '''
        # set current angle as target to store it
        self.target[2]=self.state[2]
        self.algostate=self.algostates.turnaround
        self.turnstate=0

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
        elif self.algostate==self.algostates.turnaround:
            self.turnaroundupdate()
        else:
            # stop
            self.movestate=self.movestates.stop
        
    def moveupdate(self):
        '''
        update motor signals according to movestate
        '''
        vel=self.veldamper*self.VEL
        if self.movestate==self.movestates.forward:
            self.lm.setVelocity(vel)
            self.rm.setVelocity(vel)
        elif self.movestate==self.movestates.backward:
            self.lm.setVelocity(-vel)
            self.rm.setVelocity(-vel)
        elif self.movestate==self.movestates.left:
            self.lm.setVelocity(-vel)
            self.rm.setVelocity(vel)
        elif self.movestate==self.movestates.right:
            self.lm.setVelocity(vel)
            self.rm.setVelocity(-vel)
        else:
            # stop
            self.lm.setVelocity(0)
            self.rm.setVelocity(0)
            self.veldamper=1.0

    def linmoveupdate(self):
        '''
        linear motion toward goal update
        '''

        poserror=np.linalg.norm(self.state[:2]-self.target[:2])
        
        if all(abs(self.state[:2]-self.target[:2])<self.EPSILON):
            # if x, y does match (on target), orient robot toward goal orientation, if there is one.
            if self.target[2]==None or abs(self.state[2]-self.target[2])<self.ANGEPSILON:
                self.movestate=self.movestates.stop
                self.algostate=self.algostates.stop
            else:
                self.rotateto(self.target[2])
        else:
            # if x, y doesn't match (not on target), orient robot toward goal.
            # calculate target angle
            targetang=np.arctan2(self.target[1]-self.state[1], self.target[0]-self.state[0])
            if abs(self.state[2]-targetang)<self.ANGEPSILON:
                # if on target angle, go forward
                self.movestate=self.movestates.forward
                # apply damper
                self.veldamper=max(min(abs(poserror)/(self.VELCUTOFF*self.EPSILON), 1), 0.2)
            else:
                # if not, orient robot
                self.rotateto(targetang)

    def turnaroundupdate(self):
        '''
        turn around update.
        does 3x120 deg rotations.
        '''

        if self.turnstate==0:
            self.rotateto((self.target[2] + 2/3*np.pi) % (2*np.pi))
            # if done
            if self.movestate==self.movestates.stop:
                self.turnstate+=1
        elif self.turnstate==1:
            self.rotateto((self.target[2] + 4/3*np.pi) % (2*np.pi))
            # if done
            if self.movestate==self.movestates.stop:
                self.turnstate+=1
        elif self.turnstate==2:
            self.rotateto(self.target[2] % (2*np.pi))
            # if done
            if self.movestate==self.movestates.stop:
                self.algostate=self.algostates.stop
                self.turnstate=0
       
                
    def rotateto(self, theta):
        '''
        rotate to given angle.

        theta: desired angle in radians.
        '''
        current=self.state[2]
        delta=(theta-current)
        if abs(delta) % (2*np.pi)<self.ANGEPSILON:
            self.movestate=self.movestates.stop
        else:
            # apply damper
            self.veldamper=max(min(abs(delta)/(self.ANGCUTOFF*self.ANGEPSILON), 1), 0.2)
            # shortest rotation direction
            # source https://math.stackexchange.com/questions/110080/shortest-way-to-achieve-target-angle
            if np.sign((delta+3*np.pi)%(2*np.pi)-np.pi)+1:
                self.movestate=self.movestates.left
            else:
                self.movestate=self.movestates.right
