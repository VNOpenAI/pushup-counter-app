from threading import Thread
import cv2
import time
from  models.pushup_or_not import PushupOrNotModel

class VideoGrabber:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):

        self.stream = cv2.VideoCapture(src)
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
        
        self.grabbed = True
        self.frame = cv2.imread("images/background.jpg")
        self.stopped = False

    def start(self):    
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                frame = None
                if self.source == "video_file":
                    if time.time() - self.last_frame_time > self.spf:
                        (self.grabbed, frame) = self.stream.read()
                        self.last_frame_time = time.time()
                else:
                    (self.grabbed, frame) = self.stream.read()
                if self.grabbed and frame is not None:
                    self.frame = frame

    def get_frame(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True

    def is_stopped(self):
        return self.stopped
