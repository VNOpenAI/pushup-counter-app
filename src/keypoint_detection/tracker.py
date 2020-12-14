import numpy as np 
import cv2

class KeypointTracker():
    def __init__(self):
        self.old_frame = None
        self.old_point = None  
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def predict(self, new_frame):
        if self.old_frame is None or self.old_point is None:
            return []
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        new_point, state, error = cv2.calcOpticalFlowPyrLK(self.old_frame, new_frame, self.old_point, None, **self.lk_params)
        self.old_point = new_point
        self.old_frame = new_frame
        return new_point 

    def update(self, new_frame, new_point):
        new_point = new_point.reshape(new_point.shape[0], 1, 2).astype(int)
        self.old_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        self.old_point = new_point
