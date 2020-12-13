import cv2
import numpy as np
import time
from .common import square_padding, resize_and_pad
from ..counter.optical_flow_counter import OpticalFlowCounter

class UIDrawer:

    def __init__(self, counter: OpticalFlowCounter):

        self.main_screen = cv2.imread("images/main-screen.png")
        self.scale = 0.5
        self.main_screen = cv2.resize(self.main_screen, None, fx=self.scale, fy=self.scale)
        self.video_frame_pos = (np.array([48, 218, 1788, 1245]) * self.scale).astype(int)
        self.counting_pos = (np.array([1894, 886, 736, 148]) * self.scale).astype(int)
        self.other_info_pos = (np.array([1914, 1068, 696, 385]) * self.scale).astype(int)

        FPS = 20
        self.second_per_frame = 1 / FPS
        self.last_frame_time = time.time()
        self.current_count = 0
        self.current_frame = self.main_screen.copy()
        self.counter = counter

    def render(self):
        draw = self.current_frame.copy()
        self.update_signal()
        self.update_count()
        return draw

    def update_signal(self):
        signal_img, peak_img = self.counter.get_debug_images()
        if signal_img is not None and peak_img is not None:
            w = self.other_info_pos[2]
            h = self.other_info_pos[3]
            x = self.other_info_pos[0]
            y = self.other_info_pos[1]
            signal_img = resize_and_pad(signal_img, (w, h // 2), padColor=0)
            self.current_frame[y:y+h//2, x:x+w] = signal_img
            peak_img = resize_and_pad(peak_img, (w, h // 2), padColor=0)
            self.current_frame[y+h//2:y+h, x:x+w] = peak_img

    def set_frame(self, img):
        w = self.video_frame_pos[2]
        h = self.video_frame_pos[3]
        x = self.video_frame_pos[0]
        y = self.video_frame_pos[1]
        img = resize_and_pad(img, (w, h), padColor=0)
        self.current_frame[y:y+h, x:x+w] = img

    def update_count(self):
        w = self.counting_pos[2]
        h = self.counting_pos[3]
        x = self.counting_pos[0]
        y = self.counting_pos[1]
        self.current_frame[y:y+h, x:x+w] = self.main_screen[y:y+h, x:x+w].copy()
        number = str(self.counter.get_count()).zfill(5)
        self.current_frame[y:y+h, x:x+w] = cv2.putText(self.current_frame[y:y+h, x:x+w], number, (80, 60), cv2.FONT_HERSHEY_PLAIN,  3.0, (0, 255, 0), 3, cv2.LINE_AA) 
