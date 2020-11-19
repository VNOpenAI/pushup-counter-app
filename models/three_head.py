import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

def landmark_loss(lamda = 10):
    def landmark_loss_func(target, pred):
        esp = 1e-5
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
        reg_loss = tf.math.reduce_mean(tf.math.reduce_mean((coor_x_t-coor_x_p)**2 + (coor_y_t-coor_y_p)**2, axis=1))
        lm_loss = tf.keras.backend.switch(
            tf.keras.backend.max(target) == -1.0 and tf.keras.backend.min(target) == -1.0,
            0.0,
            angle_loss + lamda * reg_loss
        )
        return lm_loss
    return landmark_loss_func

class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
        for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

class ThreeHeadModel():

    def __init__(self, model_path, img_size=(224, 224)):
        self.model = load_model(model_path, custom_objects={"landmark_loss_func": landmark_loss(), 'FixedDropout':FixedDropout})
        self.img_size = img_size

    def predict(self, origin_img):
        origin_img = cv2.resize(origin_img, self.img_size)
        pred_landmark, pred_is_pushing_up, pred_contains_person = self.predict_batch(np.array([origin_img]))
        return pred_landmark[0], pred_is_pushing_up[0], pred_contains_person[0]

    def predict_batch(self, face_imgs, verbose=1, normalize=True):
        if normalize:
            img_batch = self.normalize_img_batch(face_imgs)
        else:
            img_batch = np.array(face_imgs)
        pred_landmark, pred_is_pushing_up, pred_contains_person = self.model.predict(img_batch, batch_size=1, verbose=verbose)
        return pred_landmark, pred_is_pushing_up, pred_contains_person

    def normalize_img_batch(self, face_imgs):
        image_batch = np.array(face_imgs, dtype=np.float32)
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