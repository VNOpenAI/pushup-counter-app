import time
import tkinter as tk
from threading import Lock, Thread
from tkinter import filedialog, messagebox, simpledialog

import cv2
import imutils
from .common import is_int

root = tk.Tk()
root.withdraw()

class VideoGrabber:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, max_width=None):

        self.max_width = max_width
        self.read_lock = Lock()
        self.default_frame = cv2.imread("images/background.jpg")
        self.frame = self.default_frame
        self.grabbed = True
        self.stopped = False
        self.open_stream(src)


    def open_stream(self, video_path:str):

        self.grabbed = False
        self.stopped = True

        # Check source
        if isinstance(video_path, int) or is_int(video_path):
            video_path = int(video_path)
            self.source = "webcam"
        elif video_path.startswith("http"):
            self.source = "webcam"
        else:
            self.source = "video_file"

        self.stream = cv2.VideoCapture(video_path)
        if not self.stream.isOpened():
            messagebox.showerror("Error", "Could not read from source: {}".format(video_path))
            return

        if self.source == "video_file":
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            self.spf = 1 / self.fps
            self.last_frame_time = time.time()

        self.grabbed = True
        self.stopped = False

    def choose_new_file(self):
        file_path = filedialog.askopenfilename()
        if file_path != "":
            self.open_stream(file_path)

    def open_camera(self):
        answer = simpledialog.askstring("Input", "Please input camera source. Put 0 for the default webcam.", parent=root, 
            initialvalue="http://192.168.43.1:8080/video")
        if answer != "":
            self.open_stream(answer)

    def start(self):    
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        while True:
            if self.stopped:
                time.sleep(1)
                continue
            frame = None
            grabbed = False
            if self.source == "video_file":
                if time.time() - self.last_frame_time > self.spf:
                    grabbed, frame = self.stream.read()
                    self.last_frame_time = time.time()
                else:
                    continue
            else:
                grabbed, frame = self.stream.read()
            
            self.read_lock.acquire()
            self.grabbed = grabbed
            self.read_lock.release()
            if frame is not None:
                if self.max_width is not None:
                    frame = imutils.resize(frame, width=self.max_width)
                self.read_lock.acquire()
                self.frame = frame
                self.read_lock.release()


    def get_frame(self):
        self.read_lock.acquire()
        frame = self.frame.copy() if self.frame is not None else self.default_frame
        self.read_lock.release()
        return frame

    def stop(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped
