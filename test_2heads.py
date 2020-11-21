import cv2
import numpy as np
from utils import common
from  models.two_head import TwoHeadModel

model_path = "data/models/2heads/efficientnetb2_2head_angle_ep150.h5"
test_video_path = "test_data/154.mp4"
net_input_size = (224, 224)

model = TwoHeadModel(model_path=model_path, img_size=net_input_size)

cap = cv2.VideoCapture(test_video_path)
if cap is None:
  print("Error reading video", test_video_path)
  exit(1)

ret, frame = cap.read()
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while ret:

    points, is_pushing_up = model.predict(frame)

    draw = frame.copy()
    for point in points:
        draw = cv2.circle(draw, (point[0], point[1]), 4, (0, 255, 0), -1)
    
    cv2.putText(draw, 'Pushing:{}'.format(is_pushing_up), (100, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 255), 1, cv2.LINE_AA) 

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(draw, [pts], True, (0,0,255), 3)

    cv2.imshow("Result", draw)
    cv2.waitKey(1)

    ret, frame = cap.read()