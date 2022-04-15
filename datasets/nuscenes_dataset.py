import os
import random
import bisect
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

from .mono_dataset import MonoDataset

import torch
from torchvision import transforms

# TODO:
#    1. feed a NuScenes instance to the constructor of NuScenesDataset
#    2. make masks with bbox annotations
#    3. make iterators in NuScenesProcessor
#    4. screen out daytime, nighttime, all
# mapping of each camera to the radars having overlap of FOV with it

class NuScenesDataset(MonoDataset):
    """ nuScenes dataset loader """
    
    def __init__(self, *args, **kwargs):
        """
        filenames ==> tokens of camera sample_data frames
        """
        super(NuScenesDataset, self).__init__(*args, **kwargs)
        self.nusc = NuScenes(version=kwargs['nuscenes_version'],
                dataroot=self.data_path)

    def check_depth(self):
        """Check if ground-truth depth exists
        set it as False for now
        """
        return False

    def __getitem__(self, index):
        """Returns a singe training data from the dataset as a dictionary

        Keys in the dictionary are either strings or tuples:

            ('token', <frame_id>)
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
        if frame_id >= 0:
            action = 'next'
        else:
            action = 'prev'

        num_tracing = abs(frame_id)

        while num_tracing > 0:
            token = self.nusc.get('sample_data', token)[action]
            num_tracing -= 1

        sample_data = self.nusc.get('sample_data', token)
        img_path = os.path.join(self.data_path, sample_data['filename'])

        return token, self.get_image(self.loader(img_path), do_flip, crop_offset)

    def load_intrinsics(self, token):
        """Returns a 4x4 camera intrinsics matrix corresponding to the token"""
        sample_data = self.nusc.get('sample_data', token)
        camera_calibration = self.nusc.get(
                'calibrated_sensor', sample_data['calibrated_sensor_token'])
        K = np.array(camera_calibration['camera_intrinsic'])
        K = np.concatenate( (K, np.array([[0,0,0]]).T), axis = 1 )
        K = np.concatenate( (K, np.array([[0,0,0,1]])), axis = 0 )
        return np.float32(K)

    def get_sensor_map(self, token, ratio, delta_u, delta_v, do_flip,
            sensor_type='radar'):
        """Obtains a depth map whose shape is consistent with the resized images
        Returns:
            a sensor map(from radars or lidar) has shape of (width, height)
        """

        camera_sample_data = self.nusc.get('sample_data', token)
        camera_channel = camera_sample_data['channel']
        img_height = camera_sample_data['height']
        img_width = camera_sample_data['width']
        #find representative frames of the radars defined by CAM2RADARS
        matched_radar_frames = self.match_dist_sensor_frames(
                camera_sample_data, sensor_type=sensor_type)

        #project radar points to images
        #concate radar maps of all the radars
        for idx, mrf in enumerate(matched_radar_frames):
            
            points, depths, _ = (
                    self.nusc.explorer.map_pointcloud_to_image(
                        mrf['token'],
                        camera_sample_data['token'])
                    )
            points[2] = depths

            if idx == 0:
                point_cloud_uv = points
            else:
                point_cloud_uv = np.concatenate((point_cloud_uv, points), axis=1)

        # adjust the projected coordinates by ratio, delta_u, delta_v
        point_cloud_uv[:2] *= ratio
        point_cloud_uv[0] -= delta_u
        point_cloud_uv[1] -= delta_v
        point_cloud_uv = point_cloud_uv[:, point_cloud_uv[0] > 0]
        point_cloud_uv = point_cloud_uv[:, point_cloud_uv[0] < self.width]
        point_cloud_uv = point_cloud_uv[:, point_cloud_uv[1] > 0]
        point_cloud_uv = point_cloud_uv[:, point_cloud_uv[1] < self.height]

        # convert to a depth map with the same shape with images
        depth_map = self.make_depthmap(point_cloud_uv, (self.height, self.width))

        if do_flip:
            depth_map = np.flip(depth_map, axis = 1)

        return depth_map

    def make_depthmap(self, point_cloud_uv, img_shape):
        """Reshape projected point cloud to a image-like map
        Args:
            point_cloud_uv(numpy.ndarray):
            img_shape(tuple): (height, width)
        """
        xs, ys, zs = point_cloud_uv
        depth_map = np.zeros(img_shape)
        depth_map[np.clip(ys.astype(np.int), 0, img_shape[0]-1),
                  np.clip(xs.astype(np.int), 0, img_shape[1]-1)] = zs

        return depth_map

    def match_dist_sensor_frames(self, camera_sample_data, sensor_type='radar'):
        """Returns the matched radar frames from the radar channels
        Args:
            sensor_type(str): 'radar' or 'lidar'
        """
        # define a binary search function only in this method frame
        # search the frame whose timestamp is closest to the camera frame
        # call get_sensor_frames_per_keyframe for each radar channel
        
        sample_token = camera_sample_data['sample_token']
        camera_channel = camera_sample_data['channel']
        camera_timestamp = camera_sample_data['timestamp']
        if sensor_type == 'radar':
            sensor_channels = CAM2RADARS[camera_channel]
        elif sensor_type == 'lidar':
            sensor_channels = ['LIDAR_TOP']

        def match(frames, target_timestamp):
            """Returns the index of the frame closest to the target_timestamp"""

            tss = [frame['timestamp'] for frame in frames]

            idx = bisect.bisect_left(tss, target_timestamp)

            if idx == 0:
                return idx
            elif idx == len(tss):
                return idx-1
            
            return np.argmin(
                    [target_timestamp - tss[idx-1], tss[idx] - target_timestamp]
                    )

        matched_frames = []

        for sensor_ch in sensor_channels:
            sensor_frames = self.get_sensor_frames_per_keyframe(
                    sample_token, sensor_ch)
            matched_idx = match(sensor_frames, camera_timestamp)
            matched_frames.append(sensor_frames[matched_idx])
        return matched_frames

    def get_sensor_frames_per_keyframe(self, sample_token, sensor):
        """
        """
        # obtain the refrence keyframe
        keyframe = self.nusc.get('sample', sample_token)

        # collecting sample_data frames synchronized by the keyframe
        sample = self.nusc.get('sample_data', keyframe['data'][sensor])
        assert sample['is_key_frame']

        sensor_frames = [sample]
        while sample['next']:
            sample = self.nusc.get('sample_data', sample['next'])
            if not sample['is_key_frame']:
                sensor_frames.append(sample)
            else:
                break

        return sensor_frames
        




