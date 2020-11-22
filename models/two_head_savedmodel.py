import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
from utils.common import square_padding

class TwoHeadModel():

    def __init__(self, model_path, img_size=(224, 224)):
        self.model = tf.saved_model.load(model_path)
        self.infer = self.model.signatures["serving_default"]
        self.infer(tf.constant(np.zeros((1, img_size[1], img_size[0], 3), dtype=np.float32), ))
        
        self.img_size = img_size

    def predict(self, origin_img):
        batch_landmarks, batch_is_pushing_up = self.predict_batch(np.array([origin_img]))
        return batch_landmarks[0], batch_is_pushing_up[0]

    def predict_batch(self, imgs, verbose=1, normalize=True):
        imgs, original_img_sizes, paddings = self.preprocessing(imgs)
        results = self.infer(tf.constant(imgs))["landmarks"].numpy()
        batch_landmarks, batch_is_pushing_up = self.postprocessing(results, paddings=paddings, original_img_sizes=original_img_sizes)
        return batch_landmarks, batch_is_pushing_up

    def postprocessing(self, results, paddings=None, original_img_sizes=None):
        batch_landmarks = results[..., :14].copy()
        batch_landmarks = batch_landmarks.reshape((-1, 7, 2))
        
        for i in range(len(batch_landmarks)):

            if paddings is not None:
                top, left, bottom, right = paddings[i]
                scale_x =  1.0 / (1 - left - right)
                scale_y =  1.0 / (1 - top - bottom)
                scale = np.array([scale_x, scale_y], dtype=np.float32)
                offset = np.array([left, top], dtype=np.float32)
                batch_landmarks[i] -= offset
                batch_landmarks[i] = batch_landmarks[i] * scale

            img_size = None
            if original_img_sizes is None:
                img_size = np.array(self.img_size)
            else:
                img_size = np.array(original_img_sizes[i])
            
            batch_landmarks[i] = batch_landmarks[i] * img_size

        batch_is_pushing_up = results[..., 14].copy()
        return batch_landmarks, batch_is_pushing_up

    def preprocessing(self, imgs):
        original_img_sizes = []
        paddings = []

        image_batch = []
        for i in range(len(imgs)):
            img_size = (imgs[i].shape[1], imgs[i].shape[0])
            original_img_sizes.append(img_size)
            img, padding = square_padding(imgs[i], desired_size=max(self.img_size), return_padding=True)
            paddings.append(padding)
            img = cv2.resize(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_batch.append(img)

        image_batch = np.array(image_batch, dtype=np.float32)
        image_batch /= 255.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_batch[..., :] -= mean
        image_batch[..., :] /= std


        return image_batch, original_img_sizes, paddings