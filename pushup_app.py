import threading
import time

import cv2
import numpy as np

from models.keypoint_heatmap import KeypointHeatmapModel
from utils.video_grabber import VideoGrabber

keypoint_model_path = "data/models/keypoint/epoch5.pt"
test_video_path = "test_data/154.mp4"
video_grabber = VideoGrabber(test_video_path).start()

model = KeypointHeatmapModel(keypoint_model_path, img_size=(225, 225))

def heatmap_thread(video_grabber, model, points_arr):
    while True:
        if video_grabber is not None:
            points = model.predict(video_grabber.get_frame())
            points_arr[0] = points
        else:
            time.sleep(100)

points_arr = [[]]
t_heatmap = threading.Thread(target=heatmap_thread, args=(video_grabber, model, points_arr))
t_heatmap.start()


cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while not video_grabber.is_stopped():
    
    draw = video_grabber.get_frame()
    points = points_arr[0]
    for point in points:
        x, y = tuple(point)
        draw = cv2.circle(draw, (x, y), 4, (0, 255, 0), -1)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(draw, [pts], True, (0,0,255), 3)

    cv2.imshow("Result", draw)
    cv2.waitKey(1)
