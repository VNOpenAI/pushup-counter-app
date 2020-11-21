from threading import Thread, Lock
import cv2
import time
import imutils
from  models.pushup_or_not import PushupOrNotModel

class VideoGrabber:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0, max_width=None):

        self.max_width = max_width
        self.stream = cv2.VideoCapture(src)
        self.read_lock = Lock()
        if self.stream is None:
            print("Could not read from source:", src)
            exit(1)
        if isinstance(src, int):
            self.source = "webcam"
        else:
            self.source = "video_file"
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            self.spf = 1 / self.fps
            self.last_frame_time = time.time()
        
        self.default_frame = cv2.imread("images/background.jpg")
        self.frame = self.default_frame
        self.grabbed = True
        self.stopped = False

    def start(self):    
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        while not self.stopped:
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
