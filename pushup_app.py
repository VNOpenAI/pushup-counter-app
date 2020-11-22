# https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/56451135#56451135

import threading
import time

import cv2
import numpy as np

from models.two_head import TwoHeadModel
from utils.video_grabber import VideoGrabber
from counters.optical_flow_counter import OpticalFlowCounter
from counters.keypoint_based_counter import KeypointBasedCounter
from utils.ui_drawer import UIDrawer

keypoint_model_path = "data/models/2heads/efficientnetb2_2head_angle_ep030.h5"
test_video_path = "test_data/154.mp4"
# test_video_path = "http://192.168.43.1:8080/video"

video_grabber = VideoGrabber(test_video_path, max_width=224).start()
# counter = OpticalFlowCounter(video_grabber, sample_time=0.05).start()
kp_counter = KeypointBasedCounter()
ui_drawer = UIDrawer(kp_counter)


model = TwoHeadModel(keypoint_model_path, img_size=(224, 224))
def keypoint_thread(video_grabber, model, points_arr):
    while True:
        if video_grabber is not None:
            frame = video_grabber.get_frame()
            points, is_pushing_up = model.predict(frame)
            points_arr[0] = points
            points_arr[1] = is_pushing_up
points_arr = [[], 0]
keypoint_t = threading.Thread(target=keypoint_thread, args=(video_grabber, model, points_arr))
keypoint_t.daemon = True
keypoint_t.start()


cv2.namedWindow("PushUp App", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("PushUp App", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    
    video_frame = video_grabber.get_frame()
    points = points_arr[0]
    is_pushing_up = points_arr[1]
    for point in points:
        x, y = tuple(point)
        draw = cv2.circle(video_frame, (x, y), 4, (0, 255, 0), -1)

    kp_counter.update_points(points)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(video_frame, [pts], True, (0,0,255), 3)

    cv2.putText(video_frame, str(is_pushing_up), (10, 50), cv2.FONT_HERSHEY_COMPLEX,  
                   0.4, (0,0,255), 1, cv2.LINE_AA)

    ui_drawer.set_frame(video_frame)
    draw = ui_drawer.render()

    cv2.imshow("PushUp App", draw)
    
    k = cv2.waitKey(30)
    k = k & 0xFF
    if k == ord("o"):
        video_grabber.choose_new_file()
    elif k == ord("c"):
        video_grabber.open_camera()
    elif k == ord("q"):
        exit(0)
    
    # if cv2.getWindowProperty("PushUp App", cv2.WND_PROP_VISIBLE) < 0:
    #     exit(0)
