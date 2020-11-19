import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin
from imutils import face_utils

def get_list_from_filenames(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def normalize_landmark_point(original_point, image_size):
    '''
    original_point: (x, y)
    image_size: (W, H)
    '''
    x, y = original_point
    x -= image_size[0] // 2
    y -= image_size[1] // 2
    x /= image_size[0]
    y /= image_size[1]
    return [x, y]

def unnormalize_landmark_point(normalized_point, image_size, scale=[1,1]):
    '''
    normalized_point: (x, y)
    image_size: (W, H)
    '''
    x, y = normalized_point
    x *= image_size[0]
    y *= image_size[1]
    x += image_size[0] // 2
    y += image_size[1] // 2
    x *= scale[0]
    y *= scale[1]
    return [x, y]

def unnormalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.multiply(np.array(landmark), np.array(image_size)) 
    landmark = landmark + image_size / 2
    return landmark

def normalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.array(landmark) - image_size / 2
    landmark = np.divide(landmark, np.array(image_size))
    return landmark

def draw_landmark(img, landmark):
    im_width = img.shape[1]
    im_height = img.shape[0]
    img_size = (im_width, im_height)
    landmark = landmark.reshape((-1, 2))
    unnormalized_landmark = unnormalize_landmark(landmark, img_size)
    for i in range(unnormalized_landmark.shape[0]):
        img = cv2.circle(img, (int(unnormalized_landmark[i][0]), int(unnormalized_landmark[i][1])), 2, (0,255,0), 2)
    return img


def crop_loosely(shape, img, input_size, landmark=None):
    bbox, scale_x, scale_y = get_loosen_bbox(shape, img, input_size)
    crop_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    crop_face = cv2.resize(crop_face, input_size)
    return crop_face

def get_loosen_bbox(shape, img, input_size):
    max_x = min(shape[2], img.shape[1])
    min_x = max(shape[0], 0)
    max_y = min(shape[3], img.shape[0])
    min_y = max(shape[1], 0)
    
    Lx = max_x - min_x
    Ly = max_y - min_y
    
    Lmax = int(max(Lx, Ly) * 2.0)
    
    delta = Lmax * 0.4
    
    center_x = (shape[2] + shape[0]) // 2
    center_y = (shape[3] + shape[1]) // 2
    start_x = int(center_x - delta)
    start_y = int(center_y - delta - 10)
    end_x = int(center_x + delta)
    end_y = int(center_y + delta - 10)
    
    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0
    if end_x > img.shape[1]:
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        end_y = img.shape[0]

    scale_x = float(input_size[0]) / (end_x - start_x)
    scale_y = float(input_size[1]) / (end_y - start_y)
    return (start_x, start_y, end_x, end_y), scale_x, scale_y