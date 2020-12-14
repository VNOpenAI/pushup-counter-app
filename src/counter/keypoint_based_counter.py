import time
import math
from threading import Thread, Lock

import cv2
import numpy as np
from ..utils.common import plot_signal
from ..utils.video_grabber import VideoGrabber
from .signal_processing import *
from .find_peaks_running import RealtimePeakDetector

class KeypointBasedCounter:

    def __init__(self, max_seq_len=200):
        """
        Args:
            video_grabber (VideoGrabber)
        """

        self.count = 0
        self.angle_seq = [0] * max_seq_len
        self.magnitude_seq = [0] * max_seq_len
        self.max_seq_len = max_seq_len
        self.peaks = [0] * max_seq_len
        self.rt_peak_finder = RealtimePeakDetector(60, 10, 0.1)
        self.prev_peak_value = 0
        self.debug_lock = Lock()

        self.signal_img = None
        self.peak_img = None
        self.max_value = 1

        self.counting = True

    def update_points(self, points):

        new_data_point = 0
        if len(points) > 0:
            # new_data_point = cv2.contourArea(np.array(points).astype(np.float32))
            new_data_point = points[3][1]
        if new_data_point > self.max_value:
            self.max_value = new_data_point

        is_peak = self.rt_peak_finder.thresholding_algo(new_data_point)
        if self.prev_peak_value == 0 and is_peak == 1:
            self.increase_count()
        self.prev_peak_value = is_peak
        self.peaks.append(is_peak)
        self.peaks = self.strip_arr(self.peaks)
        self.signals = self.rt_peak_finder.y

        self.debug_lock.acquire()

        scaled_points = np.array(self.strip_arr(self.signals)) / self.max_value * 180
        signal_img = plot_signal(scaled_points, 0, 180)
        self.signal_img = signal_img

        peak_img = plot_signal(self.strip_arr(self.peaks), 0, 1)
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
        if self.counting:
            self.count += 1

    def set_counting(self, counting):
        self.counting = counting

    def get_count(self):
        return self.count

    def reset(self):
        self.count = 0

    def preprocess(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
