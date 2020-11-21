# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/56451135#56451135
import threading
import time

import cv2
import numpy as np

from models.two_head import TwoHeadModel
from utils.video_grabber import VideoGrabber
from counters.optical_flow_counter import OpticalFlowCounter
from utils.ui_drawer import UIDrawer

keypoint_model_path = "data/models/2heads/efficientnetb2_2head_ep030.h5"
test_video_path = "test_data/154.mp4"
# test_video_path = "http://192.168.43.1:8080/video"

video_grabber = VideoGrabber(test_video_path, max_width=224).start()
counter = OpticalFlowCounter(video_grabber, sample_time=0.05).start()

ui_drawer = UIDrawer(counter)

def keypoint_thread(video_grabber, points_arr):
    model = TwoHeadModel(keypoint_model_path, img_size=(224, 224))
    while True:
        if video_grabber is not None:
            frame = video_grabber.get_frame()
            points, is_pushing_up = model.predict(frame)
            points_arr[0] = points
points_arr = [[]]
keypoint_t = threading.Thread(target=keypoint_thread, args=(video_grabber, points_arr))
keypoint_t.daemon = True
keypoint_t.start()


cv2.namedWindow("PushUp App", 0)
while True:
    
    video_frame = video_grabber.get_frame()
    points = points_arr[0]
    for point in points:
        x, y = tuple(point)
        draw = cv2.circle(video_frame, (x, y), 4, (0, 255, 0), -1)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(video_frame, [pts], True, (0,0,255), 3)

    ui_drawer.set_frame(video_frame)
    draw = ui_drawer.render()

    cv2.imshow("PushUp App", draw)
    
    k = cv2.waitKey(1)
    k = k & 0xFF
    if k == ord("o"):
        video_grabber.choose_new_file()
    elif k == ord("c"):
        video_grabber.open_camera()
