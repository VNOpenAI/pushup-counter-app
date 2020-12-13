import cv2
from src.action_recognition.push_up_recognizer import PushUpRecognizer

# model_path = "trained_models/action_recognition/chinh-mobilenetv2-2020-12-10.h5"
model_path = "trained_models/action_recognition/model_ep005.h5"
test_video_path = 0

pushup_recognizer = PushUpRecognizer(model_path)

cap = cv2.VideoCapture(test_video_path)
if cap is None:
  print("Error reading video", test_video_path)
  exit(1)

ret, frame = cap.read()
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
while ret:

    # frame = preprocess_img(frame)

    is_pushing_up, raw_score = pushup_recognizer.update_frame(frame, return_raw_score=True)

    is_pushing_up = raw_score

    points = []
    draw = frame.copy()

    text = "Pushing {}".format(is_pushing_up)
    color = (0, 255, 0)
    if is_pushing_up < 0.95:
        text = "Not Pushing {}".format(is_pushing_up)
        color = (0, 0, 255)

    cv2.putText(draw, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, color, 1, cv2.LINE_AA) 
    cv2.putText(draw, "Raw score: {}".format(raw_score), (100, 200), cv2.FONT_HERSHEY_SIMPLEX,  
                0.5, color, 1, cv2.LINE_AA) 
    cv2.imshow("Result", draw)
    cv2.waitKey(1)

    ret, frame = cap.read()