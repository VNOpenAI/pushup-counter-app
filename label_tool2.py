import requests 
import cv2
import easygui
import pathlib
import os
import json
# import ctypes

# # Query DPI Awareness (Windows 10 and 8)
# awareness = ctypes.c_int()
# errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
# print(awareness.value)

# video_id = None

# # Set DPI Awareness  (Windows 10 and 8)
# errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
# # the argument is the awareness level, which can be 0, 1 or 2:
# # for 1-to-1 pixel control I seem to need it to be non-zero (I'm using level 2)

SERVER_URL = "https://vinbdi-label.herokuapp.com"
VIDEO_BASE_URL = "https://pushup.imfast.io"
LOCAL_VIDEO_FOLDER = "videos"
pathlib.Path(LOCAL_VIDEO_FOLDER).mkdir(parents=True, exist_ok=True)
friend_name = "unknown"

cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)

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

def end_program():
    easygui.msgbox("All tasks were done. Goodbye!", title="Close")
    exit(0)

def fetch_new_task():
    """Fetch new task from server"""
    task_url = SERVER_URL + "/get_video_to_label"
    try:
        r = requests.get(url = task_url) 
        data = r.json()["result"]
        if not data.get("success", False):
            easygui.msgbox(data.get("message", "Unknown error. Sorry!"), title="Oh no.")
        return data["your_task"]
    except:
        easygui.msgbox("Unknown error. Sorry!", title="Oh no.")
    return None


def download_file(url, file_path):
    """Download from server"""
    myfile = requests.get(url)
    open(file_path, 'wb').write(myfile.content)
    print("Downloaded video file to", file_path)

def submit_label(video_id, label):
    """Submit a label to server"""
    submit_url = SERVER_URL + "/labels"
    if not label:
        return
    label = {"label": label, "labler": friend_name}
    r = requests.post(submit_url, json={"video_id": video_id, "label": json.dumps(label)})
    print(r.content)
    data = r.json()
    if not data.get("success", False):
        easygui.msgbox(data.get("message", "Unknown error. Sorry!"), title="Oh no.")


labels = []
def add_label_point(event,x,y,flags,param):
    global labels
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        labels.append([mouseX, mouseY])
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(labels) > 0:
            labels.pop()

def next_frames(video, n):
    global frame, frame_id, last_valid_frame
    global labels
    for _ in range(n):
        ret, frame = video.read()
        if not ret:
            return False
        frame = preprocess_img(frame)
        last_valid_frame = frame
        frame_id += 1
    labels = []
    return True

def save_label(frame_id, video):
    global labels, video_id
    submit_label(video_id, {"frame_id": frame_id, "points": labels})
    next_frames(video, 10)

def label_video(video_path):
    """Label a video"""

    global frame_id, frame, last_valid_frame
    global labels

    cv2.namedWindow('Labeling', 0)
    cv2.setMouseCallback('Labeling', add_label_point)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    spf = 1 / fps
    

    if not video.isOpened():
        print("Error opening video")
    ret, frame = video.read()
    if not ret:
        return None
    frame = preprocess_img(frame)

    frame_id = 0
    count = 0
    frame_ids = []
    last_valid_frame = None
    
    next_frames(video, 10)

    while True:

        draw = last_valid_frame.copy()
        hidden_ptrs = []
        for i, label in enumerate(labels):
            if label[0] == -1:
                hidden_ptrs.append(i)
                continue
            draw = cv2.circle(draw, tuple(label), 3, (0,255,0), -1)
            draw = cv2.putText(draw,  str(i), (label[0]+10, label[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
        draw = cv2.putText(draw,  "Hidden:" + ",".join(map(str, hidden_ptrs)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA) 
        cv2.imshow('Labeling', draw)

        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            end_program()
        elif k & 0xFF == ord('r'):
            return "RESTART"
        elif k & 0xFF == 32:
            if not next_frames(video, 10):
                break
        elif k & 0xFF == ord('x'):
            labels.append([-1, -1])
        elif k & 0xFF == ord('u'):
            return "SKIP"
        elif k & 0xFF == ord('s'):
            save_label(frame_id, video)

    video.release()


def label_task():
    global video_id
    """Label 1 more image"""

    print("=== New Task")

    # Fetch a new task
    print("1. Fetching new task")
    task = fetch_new_task()
    if not task:
        end_program()

    # Download video file
    print("2. Download video file")
    video_id = task["video_id"]
    video_url = VIDEO_BASE_URL + "/" + task["video_url"]
    video_file = os.path.join(LOCAL_VIDEO_FOLDER, "{}.mp4".format(video_id))
    download_file(video_url, video_file)

    # Label video
    label = label_video(video_file)
    while label == "RESTART":
        label = label_video(video_file)
    
easygui.msgbox("Thank you very much for being here. This program was written for image labeling. Click OK to continue.", title="Hello, my friend!")
asked_friend_name = easygui.enterbox("Enter your email:")
if asked_friend_name is not None and asked_friend_name != "":
    friend_name = asked_friend_name

while True:
    label_task()