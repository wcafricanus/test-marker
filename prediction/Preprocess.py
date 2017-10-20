import numpy as np
import cv2


def render_image(points_list):
    image = draw_greyscale_digit(points_list)
    centralize_image(image)
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
    shift_x = max(0, offset_x)
    offset_y = ((height - (rect[1] + rect[3])) - rect[1]) // 2
    shift_y = max(0, offset_y)
    image[rect[1] + shift_y:rect[1] + rect[3] + shift_y, rect[0] + shift_x:rect[0] + rect[2] + shift_x] = \
        image_copy[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


def strip_label(point_list):
    result = [[item['x'], item['y']] for item in point_list]
    return result


def draw_greyscale_digit(points_list):
    # Create a black image
    img = np.zeros((162, 180, 1), np.float32)
    img.fill(0.)

    processed_array = [np.array(item, np.int32).reshape((-1, 1, 2)) for item in points_list]

    img = cv2.polylines(img, processed_array, False, 0.99, 7)
    img = cv2.resize(img, (28, 28))

    return img


def preprocess_data(data):
    result = {}
    for key in sorted([int(key) for key in data.keys()]):
        paths = data[str(key)]['pathList']
        correct_num = data[str(key)]['vasQues']
        if paths:
            points_list = [strip_label(item['pointList']) for item in paths]
            image = render_image(points_list)
            result[key] = (image, correct_num)
        else:
            result[key] = (None, correct_num)

    return result