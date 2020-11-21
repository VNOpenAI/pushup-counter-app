import cv2
import numpy as np
from utils import common
from  models.three_head import ThreeHeadModel


# Init model
model_path = "data/models/3heads/efficientb3_3head_newloss_sigmoid_ep025.h5"
test_video_path = "test_data/154.mp4"
net_input_size = (224, 224)
model = ThreeHeadModel(model_path=model_path, img_size=net_input_size)

cap = cv2.VideoCapture(test_video_path)
if cap is None:
  print("Error reading video", test_video_path)
  exit(1)

def preprocess_img(im, desired_size=800):
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im

ret, frame = cap.read()
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while ret:

    frame = preprocess_img(frame)

    scale = np.divide(np.array([frame.shape[1], frame.shape[0]]), np.array(net_input_size))
    landmark, is_pushing_up, contains_person = model.predict(frame)

    points = []
    draw = frame.copy()
    for j in range(7):
        x = landmark[2 * j]
        y = landmark[2 * j + 1]
        x, y = common.unnormalize_landmark_point(
            (x, y), net_input_size, scale=scale)
        x = int(x)
        y = int(y)
        points.append([x, y])
        draw = cv2.circle(draw, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(draw, 'Pushing:{}, Person:{}'.format(is_pushing_up, contains_person), (100, 100), cv2.FONT_HERSHEY_SIMPLEX ,  
                0.5, (0, 0, 255), 1, cv2.LINE_AA) 

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(draw, [pts], True, (0,0,255), 3)

    cv2.imshow("Result", draw)
    cv2.waitKey(1)

    ret, frame = cap.read()