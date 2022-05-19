# Copyright © 2022, Bolian Chen. Released under the MIT license.

import os
import random
import bisect
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

from .mono_dataset import pil_loader, MonoDataset

import torch
import torch.utils.data as data
from torchvision import transforms


class NuScenesDataset(MonoDataset):
    """ nuScenes dataset loader """
    
    def __init__(self, *args, **kwargs):
        """
        filenames ==> tokens of camera sample_data frames
        """
        super(NuScenesDataset, self).__init__(*args, **kwargs)
        self.nusc_proc = kwargs['proc']
        self.nusc = self.nusc_proc.get_nuscenes_obj()

    def check_depth(self):
        """Check if ground-truth depth exists
        set it as False for now
        """
        return False

    def __getitem__(self, index):
        """Returns a singe training data from the dataset as a dictionary

        Keys in the dictionary are either strings or tuples:
            ('color', <frame_id>, <scale>)
            ('color_aug', <frame_id>, <scale>)
            ('mask', <frame_id>, <scale>)
            ('radar', <frame_id>, 0)
            ('lidar', <frame_id>, 0)
            ('K', <scale>)
            ('inv_K', <scale>)

        <frame_id>: an integer representing the temporal adjacency relative to
                    the frame retrieved by 'index':
                    0: the frame itself 
                    1: its next frame
                    -1: its previouse frame
                    and so on.

        Several unsupervised depth estimation methods implement multi-scale
        reconstruction loss. The loss may need downsacled data in some cases
        
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)

        Downscaled radar and lidar data are not implemented for now

        """

        # an empty dictionary
        inputs = {}

        # do augmentation
        do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = (not self.not_do_color_aug) and do_color_aug

        # determine whether to flip images horizontally
        # used by get_color and process_intrinsics
        do_flip = self.is_train and random.random() > 0.5
        do_flip = (not self.not_do_flip) and do_flip

        # Initialize cropping method
        if self.do_crop:
            # Random crop for training and center crop for validation
            crop_offset = -1 if self.is_train else -2
        else:
            crop_offset = -3

        token = self.filenames[index]

        for i in self.frame_idxs:
            inputs[('token', i)], color_info = self.get_color(
                    token, i, do_flip, crop_offset)
            inputs[('color', i, -1)], ratio, delta_u, delta_v, crop_offset = (
                    color_info)

            if self.seg_mask != 'none':
                mask = self.get_mask(token, i, do_flip, crop_offset)[0]
                inputs[('mask', i, -1)] = mask.convert('L')
                                                        

            if self.use_radar:
                inputs[('radar', i, 0)] = self.get_sensor_map(
                        inputs[('token', i)], ratio, delta_u, delta_v,
                        do_flip, sensor_type = 'radar')
            if self.use_lidar:
                inputs[('lidar', i, 0)] = self.get_sensor_map(
                        inputs[('token', i)], ratio, delta_u, delta_v,
                        do_flip, sensor_type = 'lidar')
        # crop_offset would become the number of pixels to crop from the top

        # adjusting intrinsics to match each scale in the pyramid
        K = self.load_intrinsics(token)
        self.adjust_intrinsics(K, inputs, ratio, delta_u, delta_v, do_flip)
        
        if do_color_aug:
            # return a transform
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        

        self.preprocess(inputs, color_aug)

        # delete the images of original scale
        for i in self.frame_idxs:
            del inputs[('token', i)]
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            if self.seg_mask != 'none':
                del inputs[("mask", i, -1)]

        return inputs

    def get_color(self, token, frame_id, do_flip, crop_offset=-3):
        """Returns an resized RGB image and its camera sample_data token"""
        # get the token of the adjacent frame
        token = self.nusc_proc.get_adjacent_token(token, frame_id)
        sample_data = self.nusc.get('sample_data', token)
        img_path = os.path.join(self.data_path, sample_data['filename'])
        return token, self.get_image(self.loader(img_path), do_flip, crop_offset)

    def get_mask(self, token, frame_id, do_flip, crop_offset=-3):
        """Return an Resized segmentation mask
        """
        token = self.nusc_proc.get_adjacent_token(token, frame_id)
        mask = self.nusc_proc.get_seg_mask(token)
        return self.get_image(mask, do_flip, crop_offset, inter=cv2.INTER_NEAREST)

    def load_intrinsics(self, token):
        """Returns a 4x4 camera intrinsics matrix corresponding to the token
        """
        # 3x3 camera matrix
        K = self.nusc_proc.get_cam_intrinsics(token)
        K = np.concatenate( (K, np.array([[0,0,0]]).T), axis = 1 )
        K = np.concatenate( (K, np.array([[0,0,0,1]])), axis = 0 )
        return np.float32(K)

    def get_sensor_map(self, cam_token, ratio, delta_u, delta_v, do_flip,
            sensor_type='radar'):
        """Obtains a depth map whose shape is consistent with the resized images
        Args:
            cam_token: a camera sample_data token
        Returns:
            a sensor map(from radars or lidar) has shape of (width, height)
        """

        point_cloud_uv = self.nusc_proc.get_proj_dist_sensor(
                cam_token, sensor_type=sensor_type)

        point_cloud_uv = self.nusc_proc.adjust_cloud_uv(point_cloud_uv,
                self.width, self.height, ratio, delta_u, delta_v)

        # convert to a depth map with the same shape with images
        depth_map = self.nusc_proc.make_depthmap(
                point_cloud_uv, (self.height, self.width))

        if do_flip:
            depth_map = np.flip(depth_map, axis = 1)

        return depth_map
