import threading
from threading import Lock

import cv2
import numpy as np

from src.keypoint_detection.blazepose_heatmap import BlazePoseHeatmap
from src.utils.video_grabber import VideoGrabber
from src.counter.keypoint_based_counter import KeypointBasedCounter
from src.utils.ui_drawer import UIDrawer
from src.utils.visualizer import visualize_keypoints
from src.action_recognition.push_up_recognizer import PushUpRecognizer

keypoint_model_path = "trained_models/keypoint_detection/blazepose_heatmap_v1.onnx"
is_pushup_model_path = "trained_models/action_recognition/va-mobilenetv2-2020-12-10-ep4.h5"
# test_video_path = "http://192.168.43.1:8080/video"
test_video_path = 0

keypoint_lock = Lock()

video_grabber = VideoGrabber(test_video_path, max_width=512).start()
kp_counter = KeypointBasedCounter()
ui_drawer = UIDrawer(kp_counter)
keypoint_detector = BlazePoseHeatmap(keypoint_model_path)
pushup_recognizer = PushUpRecognizer(is_pushup_model_path)

points_arr = [[], []]
is_pushup_score = 0.0


def keypoint_thread(video_grabber, keypoint_detector, points_arr):
    global is_pushup_score
    while True:
        if video_grabber is not None:
            frame = video_grabber.get_frame()
            points = keypoint_detector.detect_keypoints(frame)
            points = np.array(points)
            visibility = np.logical_and(points[:, 0] > 0, points[:, 1] > 0)
            keypoint_lock.acquire()
            points_arr[0] = points[:, :2]
            points_arr[1] = visibility
            keypoint_lock.release()
            _, is_pushup_score = pushup_recognizer.update_frame(frame, return_raw_score=True)
keypoint_t = threading.Thread(target=keypoint_thread, args=(video_grabber, keypoint_detector, points_arr))
keypoint_t.daemon = True
keypoint_t.start()


cv2.namedWindow("PushUp App", cv2.WINDOW_NORMAL)
# cv2.setWindowProperty("PushUp App", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while True:
    
    video_frame = video_grabber.get_frame()
    keypoint_lock.acquire()
    points, visibility = points_arr
    keypoint_lock.release()

    points = np.array(points).astype(int).tolist()
    for point in points:
        x, y = tuple(point)
        draw = cv2.circle(video_frame, (x, y), 4, (0, 255, 0), -1)

    kp_counter.update_points(points)

    video_frame = visualize_keypoints(video_frame, points, visibility=visibility, edges=[[0,1,2,3,4,5,6]],
        point_color=(0,0,255), text_color=(0,255,0))

    text = "Pushing {}".format(is_pushup_score)
    color = (0, 255, 0)
    if is_pushup_score < 0.90:
        text = "Not Pushing {}".format(is_pushup_score)
        color = (0, 0, 255)
    cv2.putText(video_frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, color, 1, cv2.LINE_AA) 

    ui_drawer.set_frame(video_frame)
    draw = ui_drawer.render()

    cv2.imshow("PushUp App", draw)
    
    k = cv2.waitKey(30)
    k = k & 0xFF
    if k == ord("o"):
        video_grabber.choose_new_file()
        keypoint_lock.acquire()
        points_arr = [[], []]
        keypoint_lock.release()
        kp_counter.reset()
    elif k == ord("c"):
        video_grabber.open_camera()
        keypoint_lock.acquire()
        points_arr = [[], []]
        keypoint_lock.release()
        kp_counter.reset()
    elif k == ord("q"):
        exit(0)
