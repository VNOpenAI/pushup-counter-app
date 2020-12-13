import os
import io
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin


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
    # x -= image_size[0] // 2
    # y -= image_size[1] // 2
    x /= image_size[0]
    y /= image_size[1]
    return [x, y]


def unnormalize_landmark_point(normalized_point, image_size, scale=[1, 1]):
    '''
    normalized_point: (x, y)
    image_size: (W, H)
    '''
    x, y = normalized_point
    x *= image_size[0]
    y *= image_size[1]
    # x += image_size[0] // 2
    # y += image_size[1] // 2
    x *= scale[0]
    y *= scale[1]
    return [x, y]


def unnormalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.multiply(np.array(landmark), np.array(image_size))
    # landmark = landmark + image_size / 2
    return landmark


def normalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    # landmark = np.array(landmark) - image_size / 2
    landmark = np.divide(landmark, np.array(image_size))
    return landmark


def draw_landmark(img, landmark):
    im_width = img.shape[1]
    im_height = img.shape[0]
    img_size = (im_width, im_height)
    landmark = landmark.reshape((-1, 2))
    unnormalized_landmark = unnormalize_landmark(landmark, img_size)
    for i in range(unnormalized_landmark.shape[0]):
        img = cv2.circle(img, (int(unnormalized_landmark[i][0]), int(
            unnormalized_landmark[i][1])), 2, (0, 255, 0), 2)
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


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_signal(x, min_val, max_val, peaks=None):

    x = np.array(x)
    width = 400
    height = 100
    len_x = len(x)
    y = np.array(range(len_x))
    y = y * width / len_x
    y = y.astype(int)

    if len(x) > width:
        x = x[:-width]
    x = (x - min_val) / (max_val - min_val) * height
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(1, len(x)):
        p1 = (y[i], int(height - x[i]))
        p2 = (y[i-1], int(height - x[i-1]))
        img = cv2.line(img, p1, p2, (0, 255, 0), 2)

    if peaks is not None:
        for p in peaks:
            img[:, y[p]-1:y[p]+1] = (0, 0, 255)

    return img


def square_padding(im, desired_size=800, return_padding=False):

    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    if not return_padding:
        return new_im
    else:
        h, w = new_im.shape[:2]
        padding = (top / h, left / w, bottom / h, right / w)
        return new_im, padding

def resize_and_pad(img, size, padColor=255):
    h, w = img.shape[:2]
    sw, sh = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    scaled_img = cv2.resize(scaled_img, size)

    return scaled_img


def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False