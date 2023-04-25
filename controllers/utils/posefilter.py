'''
posefilter.py - functions for filtering and fusing pose estimates
'''
import numpy as np


class PoseFilter():

    def __init__(self, camavglen):
        '''
        camavglen: how many past values to take the average of.
        '''
        self.camavglen=camavglen
        self.camvals=np.empty((camavglen,3))
        self.camvals[:]=np.nan

    def addcamestimate(self, estimate):
        '''
        add a new pose estimate sourced from the camera.

        estimate: 3-element numpy array, [x,y,theta]
        '''
        if not (estimate is None):
            self.camvals=np.roll(self.camvals,1,0)
            self.camvals[0]=estimate

    def camaverage(self):
        '''
        get the pose average of the camera-sourced estimates.

        returns: 3-element numpy array, [x,y,theta]
        '''
        return np.nanmean(self.camvals,0)
