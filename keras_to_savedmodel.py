from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.client import session
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import load_model
import numpy as np

# This line must be executed before loading Keras model.
K.set_learning_phase(0)


H5_MODEL = "data/models/2heads/efficientnetb2_2head_ep030.h5"
OUTPUT_SAVED_MODEL = "data/models/2heads/efficientnetb2_2head_ep030"

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


model = load_model(H5_MODEL, custom_objects={'FixedDropout':FixedDropout, 'landmark_loss_func': landmark_loss()})


model.save(OUTPUT_SAVED_MODEL)