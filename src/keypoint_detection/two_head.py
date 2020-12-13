import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from utils.common import square_padding


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
        for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

def landmark_loss(alpha=0.8, beta=0.2):
    def landmark_loss_func(target, pred):
        coor_x_t = target[:][:,::2]
        coor_y_t = target[:,1:][:,::2]
        coor_x_p = pred[:][:,::2]
        coor_y_p = pred[:,1:][:,::2]
        ra1_t = tf.math.atan2((coor_y_t[:,1] - coor_y_t[:,0]), (coor_x_t[:,1] - coor_x_t[:,0] + 1e-5))
        ra1_p = tf.math.atan2((coor_y_p[:,1] - coor_y_p[:,0]), (coor_x_p[:,1] - coor_x_p[:,0] + 1e-5))
        ra2_t = tf.math.atan2((coor_y_t[:,2] - coor_y_t[:,1]), (coor_x_t[:,2] - coor_x_t[:,1] + 1e-5))
        ra2_p = tf.math.atan2((coor_y_p[:,2] - coor_y_p[:,1]), (coor_x_p[:,2] - coor_x_p[:,1] + 1e-5))
        la1_t = tf.math.atan2((coor_y_t[:,-2] - coor_y_t[:,-1]), (coor_x_t[:,-2] - coor_x_t[:,-1] + 1e-5))
        la1_p = tf.math.atan2((coor_y_p[:,-2] - coor_y_p[:,-1]), (coor_x_p[:,-2] - coor_x_p[:,-1] + 1e-5))
        la2_t = tf.math.atan2((coor_y_t[:,-3] - coor_y_t[:,-2]), (coor_x_t[:,-3] - coor_x_t[:,-2] + 1e-5))
        la2_p = tf.math.atan2((coor_y_p[:,-3] - coor_y_p[:,-2]), (coor_x_p[:,-3] - coor_x_p[:,-2] + 1e-5))
        angle_loss = tf.math.reduce_mean(((ra1_t - ra1_p)/(8*np.pi))**2+((ra2_t - ra2_p)/(8*np.pi))**2+((la1_t - la1_p)/(8*np.pi))**2+((la2_t - la2_p)/(8*np.pi))**2)
        bce_loss = tf.keras.losses.binary_crossentropy(target, pred)
        lm_loss = alpha * bce_loss + beta * angle_loss
        return lm_loss
    return landmark_loss_func

class TwoHeadModel():

    def __init__(self, model_path, img_size=(224, 224)):
        self.model = load_model(model_path, custom_objects={'FixedDropout':FixedDropout, 'landmark_loss_func': landmark_loss()}, compile=False)
        self.model.predict(np.zeros((1, 224, 224, 3), dtype=float))
        self.img_size = img_size

    def predict(self, origin_img):
        batch_landmarks, batch_is_pushing_up = self.predict_batch(np.array([origin_img]))
        return batch_landmarks[0], batch_is_pushing_up[0]

    def predict_batch(self, imgs, verbose=0):
        imgs, original_img_sizes, paddings = self.preprocessing(imgs)
        st = time.time()
        results = self.model.predict(imgs, batch_size=1, verbose=verbose)
        en = time.time()
        print(en-st)
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
                scale = np.array([scale_x, scale_y], dtype=float)
                offset = np.array([left, top], dtype=float)
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
