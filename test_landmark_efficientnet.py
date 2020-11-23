import cv2
import numpy as np
from  models.keypoint_heatmap import KeypointHeatmapENB2Model

keypoint_model_path = "data/models/keypoint/epoch32.pt"
test_video_path = "test_data/154.mp4"

model = KeypointHeatmapENB2Model(keypoint_model_path, img_size=(225, 225))

cap = cv2.VideoCapture(test_video_path)
if cap is None:
  print("Error reading video", test_video_path)
  exit(1)

ret, frame = cap.read()
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while ret:

    points = model.predict(frame)

    draw = frame.copy()
    for point in points:
        x, y = tuple(point)
        draw = cv2.circle(draw, (x, y), 4, (0, 255, 0), -1)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(draw, [pts], True, (0,0,255), 3)

    cv2.imshow("Result", draw)
    cv2.waitKey(1)

    ret, frame = cap.read()