import numpy as np
import cv2


def render_image(points_list, width, height):
    image = draw_greyscale_digit(points_list, width, height)
    image = rescale_image(image)
    centralize_image(image)
    return image


def rescale_image(image):
    image = np.reshape(image, (image.shape[0],image.shape[1]))
    ret, im_th = cv2.threshold(image, 0.1, 1.0, cv2.THRESH_BINARY)
    im_th_uint = im_th.astype(np.dtype('u1'))
    points = cv2.findNonZero(im_th_uint)
    rect = cv2.boundingRect(points)

    width = image.shape[1]
    height = image.shape[0]
    digit_width = rect[2]
    digit_height = rect[3]

    if digit_height/height>0.6:
        h = 0.6*height/digit_height
    else:
        h = 1
    if digit_width/width>0.6:
        w = 0.6*width/digit_width
    else:
        w = 1

    new_width = int(width*w)
    new_height = int(height*h)
    dim = (new_width, new_height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image.fill(0.)
    image[:resized.shape[0],:resized.shape[1]] = resized
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    return image


def centralize_image(image):
    ret, im_th = cv2.threshold(image, 0.1, 1.0, cv2.THRESH_BINARY)
    im_th_uint = im_th.astype(np.dtype('u1'))
    points = cv2.findNonZero(im_th_uint)
    rect = cv2.boundingRect(points)
    image_copy = np.copy(image)
    image.fill(0.)
    height, width = np.shape(image)
    offset_x = ((width - (rect[0] + rect[2])) - rect[0]) // 2
    shift_x = offset_x
    offset_y = ((height - (rect[1] + rect[3])) - rect[1]) // 2
    shift_y = offset_y
    image[rect[1] + shift_y:rect[1] + rect[3] + shift_y, rect[0] + shift_x:rect[0] + rect[2] + shift_x] = \
        image_copy[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def strip_label(point_list):
    result = [[item['x'], item['y']] for item in point_list]
    return result


def draw_greyscale_digit(points_list, width, height):
    # Create a black image
    img = np.zeros((height, width, 1), np.float32)
    img.fill(0.)

    processed_array = [np.array(item, np.int32).reshape((-1, 1, 2)) for item in points_list]

    img = cv2.polylines(img, processed_array, False, 1.0, 12)

    return img


def preprocess_data(vas_cog_block, vas_block_size):
    width = vas_block_size['width']
    height = vas_block_size['height']
    data = vas_cog_block
    result = {}
    for key in sorted([int(key) for key in data.keys()]):
        paths = data[str(key)]['path_list']
        correct_num = data[str(key)]['vas_ques']
        if paths:
            image = build_image_from_paths(paths, width, height)
            result[key] = (image, correct_num)
        else:
            result[key] = (None, correct_num)
    return result


def preprocess_single_data(vas_cog_block, vas_block_size):
    paths = vas_cog_block['path_list']
    correct_num = vas_cog_block['vas_ques']
    width = vas_block_size['width']
    height = vas_block_size['height']
    image = build_image_from_paths(paths, width, height)
    result = {0: (image, correct_num)}
    return result


def build_image_from_paths(paths, width, height):
    points_list = [strip_label(item['point_list']) for item in paths]
    image = render_image(points_list, width, height)
    return image
