import time
import math
from threading import Thread, Lock

import cv2
import numpy as np
from ..utils.common import plot_signal
from ..utils.video_grabber import VideoGrabber
from .signal_processing import *
from .find_peaks_running import RealtimePeakDetector

class OpticalFlowCounter:
    """
    Counter using optical flow
    """

    def __init__(self, video_grabber: VideoGrabber, sample_time=0.01, img_size=(256, 256), max_seq_len=200):
        """
        Args:
            video_grabber (VideoGrabber)
            counting_var (list): Counting variable - A list with an int. The counter will increase this variable overtime
            sample_time (float, optional): Duration between 2 sampling time point. Defaults to 0.05.
        """

        self.count = 0
        self.video_grabber = video_grabber
        self.sample_time = sample_time
        self.last_sample_time_point = time.time()
        self.img_size = img_size
        self.prev_frame = None
        # self.angle_seq = [0] * max_seq_len
        # self.magnitude_seq = [0] * max_seq_len
        self.max_seq_len = max_seq_len
        self.peaks = [0] * max_seq_len
        self.rt_peak_finder = RealtimePeakDetector(60, 10, 0.1)
        self.prev_peak_value = 0
        self.debug_lock = Lock()

        self.signal_img = None
        self.peak_img = None


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

        sum_magnitude = np.sum(magnitude)
        if sum_magnitude == 0:
            new_angle = 0
        else:
            new_angle = np.sum(angle * magnitude) / sum_magnitude
        if math.isnan(new_angle):
            new_angle = 0
        is_peak = self.rt_peak_finder.thresholding_algo(new_angle)
        if self.prev_peak_value == 0 and is_peak == 1:
            self.increase_count()
        self.prev_peak_value = is_peak
        self.peaks.append(is_peak)
        self.peaks = self.strip_arr(self.peaks)
        self.signals = self.rt_peak_finder.y


        self.debug_lock.acquire()

        signal_img = plot_signal(self.strip_arr(self.signals), 0, 180)
        # cv2.imshow("Angle", img)
        # cv2.waitKey(1)
        self.signal_img = signal_img

        peak_img = plot_signal(self.strip_arr(self.peaks), 0, 1)
        # cv2.imshow("is_peak", img)
        # cv2.waitKey(1)
        self.peak_img = peak_img

        self.debug_lock.release()

    def get_debug_images(self):
        self.debug_lock.acquire()
        signal_img = self.signal_img.copy() if self.signal_img is not None else None
        peak_img = self.peak_img.copy() if self.peak_img is not None else None
        self.debug_lock.release()
        return signal_img, peak_img

    def strip_arr(self, arr):
        if len(arr) > self.max_seq_len:
            arr = arr[-self.max_seq_len:]
        return arr

    def increase_count(self):
        self.count += 1

    def get_count(self):
        return self.count

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
