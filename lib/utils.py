from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2

from PIL import Image

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def image_resize(image, target_h, target_w, shift_h, shift_w,
                 inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the raw image size

    is_pil = isinstance(image, Image.Image)

    if is_pil:
        image = np.array(image)

    (raw_h, raw_w) = image.shape[:2]

    assert raw_h >= target_h, 'must be downscaling'
    assert raw_w >= target_w, 'must be downscaling'

    if target_h/raw_h <= target_w/raw_w:
        # calculate the ratio of the width and construct the dimensions
        r = target_w / float(raw_w)
        dim = (target_w, int(raw_h * r))

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]
        
        start = int(new_h*shift_h)
        end = start + target_h
       
        assert start >=0
        assert end <= new_h

        if len(image.shape) == 3:
            image = image[start:end,:,:]
        else:
            image = image[start:end,:]

        delta_u = 0
        delta_v = start  

    else: 
        # calculate the ratio of the height and construct the dimensions
        r = target_h / float(raw_h)
        dim = (int(raw_w * r), target_h)

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]

        start = int(new_w*shift_w)
        end = start + target_w
        image = image[:,start:end,:]

        assert start >=0
        assert end <= new_w

        if len(image.shape) == 3:
            image = image[:,start:end,:]
        else:
            image = image[:,start:end]

        delta_u = start
        delta_v = 0

    if is_pil:
        image = Image.fromarray(image)

    return image, r, delta_u, delta_v

def check_if_scene_pass(scene_description, pass_filters=['day', 'night', 'rain']):
    """
    Args:
        scene_description(str): description included in the scene metadata
        pass_filters(list of str): check the consequeneces below
            ['day']: daytime and not rainy scenes
            ['night']: nighttime and not rainy scenes
            ['rain']: rainy scenes on both daytime and nighttime
            ['day', 'night']: daytime, nighttime, and not rainy scenes
            ['day', 'rain']: rainy scenes on daytime
            ['night', 'rain']: rainy scenes on nighttime
            ['day', 'night', 'rain']: all the scenes
    """
    description = scene_description.lower()
    pass_filters = [fil for fil in pass_filters
            if fil in ('day', 'night', 'rain')]

    if len(pass_filters) == 0:
        raise NameError('invalid options are detected in pass_filters')

    if len(pass_filters) == 3:
        return True

    if 'day' in pass_filters:
        # ['day', 'night']
        if 'night' in pass_filters:
            if 'rain' not in description:
                return True
        # ['day', 'rain']
        elif 'rain' in pass_filters:
            if 'night' not in description and 'rain' in description:
                return True
        # ['day']
        else:
            if 'night' not in description and 'rain' not in description:
                return True

    if 'night' in pass_filters:
        # ['night', 'rain']
        if 'rain' in pass_filters:
            if 'night' in description and 'rain' in description:
                return True
        # ['night']
        else:
            if 'night' in description and 'rain' not in description:
                return True

    # ['rain']
    if pass_filters == ['rain']:
        if 'rain' in description:
            return True

    return False

