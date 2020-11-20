import cv2
import numpy as np
import utils
from  models.pushup_or_not import PushupOrNotModel

model_path = "data/models/pushup_or_not/mobilenetv2_pushup_classify_ep096.h5"
test_video_path = "test_data/355.mp4"
net_input_size = (128, 128)

model = PushupOrNotModel(model_path=model_path, img_size=net_input_size)

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

    image = preprocess_img(frame, desired_size=112)
    is_pushing_up = model.predict(image)

    points = []
    draw = frame.copy()

    text = "Pushing {}".format(is_pushing_up)
    color = (0, 255, 0)
    if is_pushing_up < 0.9999:
        text = "Not Pushing {}".format(is_pushing_up)
        color = (0, 0, 255)

    cv2.putText(draw, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, color, 1, cv2.LINE_AA) 
    cv2.imshow("Result", draw)
    cv2.waitKey(1)

    ret, frame = cap.read()