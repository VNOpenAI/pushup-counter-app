import time
import math
from threading import Thread

import cv2
import numpy as np
from utils.common import plot_signal
from utils.video_grabber import VideoGrabber
from counters.signal_processing import *
from counters.find_peaks_running import RealtimePeakDetector
class OpticalFlowCounter:
    """
    Counter using optical flow
    """

    def __init__(self, video_grabber: VideoGrabber, counting_var: list, sample_time=0.01, img_size=(256, 256), max_seq_len=200):
        """
        Args:
            video_grabber (VideoGrabber)
            counting_var (list): Counting variable - A list with an int. The counter will increase this variable overtime
            sample_time (float, optional): Duration between 2 sampling time point. Defaults to 0.05.
        """

        self.video_grabber = video_grabber
        self.sample_time = sample_time
        self.last_sample_time_point = time.time()
        self.img_size = img_size
        self.prev_frame = None
        self.angle_seq = [0] * max_seq_len
        self.magnitude_seq = [0] * max_seq_len
        self.max_seq_len = max_seq_len
        self.peaks = [0] * max_seq_len
        self.rt_reak_finder = RealtimePeakDetector(self.angle_seq, 10, 2.5, 0.4)

    def start(self):    
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):

        while True:
            if time.time() - self.last_sample_time_point >= self.sample_time:
                self.last_sample_time_point = time.time()
                frame = self.video_grabber.get_frame()
                if frame is None:
                    print("None frame")
                    continue
                frame = self.preprocess(frame)
                if self.prev_frame is not None:
                    self.count_from_frame(frame)
                self.prev_frame = frame

    def count_from_frame(self, frame):

        flow = cv2.calcOpticalFlowFarneback(self.prev_frame, frame,  
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        angle = angle * 180 / np.pi / 2

        new_angle = np.sum(angle * magnitude) / np.sum(magnitude)
        is_peak = self.rt_reak_finder.thresholding_algo(new_angle)
        self.peaks.append(is_peak)
        self.peaks = self.strip_arr(self.peaks)
        self.signals = self.rt_reak_finder.y

        img = plot_signal(self.strip_arr(self.signals), 0, 180)
        cv2.imshow("Angle", img)
        cv2.waitKey(1)

        img = plot_signal(self.strip_arr(self.peaks), 0, 1)
        cv2.imshow("is_peak", img)
        cv2.waitKey(1)


    def strip_arr(self, arr):
        if len(arr) > self.max_seq_len:
            arr = arr[-self.max_seq_len:]
        return arr

    def increase_count(self):
        self.counting_var[0] += 1

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
