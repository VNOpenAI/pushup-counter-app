import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

class PushupOrNotModel():

    def __init__(self, model_path, img_size=(224, 224)):
        self.model = load_model(model_path)
        self.img_size = img_size

    def predict(self, origin_img):
        origin_img = cv2.resize(origin_img, self.img_size)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        pred_is_pushing_up = self.predict_batch(np.array([origin_img]))
        return pred_is_pushing_up[0]

    def predict_batch(self, face_imgs, verbose=1, normalize=True):
        if normalize:
            img_batch = self.normalize_img_batch(face_imgs)
        else:
            img_batch = np.array(face_imgs)
        pred_is_pushing_up = self.model.predict(img_batch, batch_size=1, verbose=verbose)
        return pred_is_pushing_up

    def normalize_img_batch(self, images):
        image_batch = np.array(images, dtype=np.float32)
        image_batch /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_batch[..., 0] -= mean[0]
        image_batch[..., 1] -= mean[1]
        image_batch[..., 2] -= mean[2]
        image_batch[..., 0] /= std[0]
        image_batch[..., 1] /= std[1]
        image_batch[..., 2] /= std[2]
        return image_batch