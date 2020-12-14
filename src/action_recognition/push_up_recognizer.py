import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .kalman_filter import SimpleKalmanFilter


class PushUpRecognizer():
    
    def __init__(self, model_path):

        self.model = tf.keras.models.load_model(
            model_path, compile=False)
        self.filter = SimpleKalmanFilter(2, 2, 2, 0.5)
        self.last_score_raw = 0.5
        self.current_score_raw = 0
        self.current_score = 0


    def preprocess_images(self, images):
        # Convert color to RGB
        for i in range(images.shape[0]):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float)
        images = np.array(images, dtype=np.float32)
        images = images / 255.0
        images -= mean
        return images

    def update_frame(self, frame, return_raw_score=False):

        frame = cv2.resize(frame, dsize=(224, 224))
        frame = np.expand_dims(frame, axis=0)
        frame = self.preprocess_images(frame)

        self.current_score_raw = self.model.predict(frame)
        self.current_score = self.filter.updateEstimate(self.last_score_raw)
        self.last_score_raw = self.current_score_raw

        if return_raw_score:
            return self.current_score, self.current_score_raw
        return self.current_score
