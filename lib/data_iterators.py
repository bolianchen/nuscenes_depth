# Copyright © 2022, Bolian Chen. Released under the MIT license.

import os
from PIL import Image
from .utils import image_resize

class NuScenesIterator:
    """An iterator to iterate over the data of the selcted scenes"""

    def __init__(self, nusc_processor, width, height, scene_names=[],
            camera_channels=['CAM_FRONT'], fused_dist_sensor='radar',
            show_bboxes=False, visibilities=['', '1', '2', '3', '4']):
        """Constructor of the iterator
        Args:
            nusc_processor: a instance of NuScenesProcessor
            width: target width of the output image
            height: target height of the output image
            scenes_names(list of str): names of the scenes to iterate
                the format of a name must be 'scene-xxxx'
                xxxx is 4 decimal digits from 0000 to 1200
            camera_channels(list of str): camera channels to show
            fused_dist_sensor(str): which distance sensor to be fused with cameras
            show_bboxes(bool): whether to display 2d bboxes on keyframes
            visibilities(list of str): visibility filter for 2d bboxes
                the higher the value the better the visibility
        """
        self.nusc_proc = nusc_processor
        self.width, self.height = width, height
        self.fused_dist_sensor = fused_dist_sensor
        self.show_bboxes = show_bboxes
        self.visibilities = visibilities

        # includes all the train scenes if no scene_names given
        if len(scene_names) == 0:
            if self.nusc_proc.get_version() != 'v1.0-test':
                self.all_camera_tokens = sum([
                    self.nusc_proc.gen_tokens(
                        is_train=True, specified_cams=camera_channels)], [])
            else:
                # TODO: speed_limits and pass_filters do not work for v1.0-test
                self.all_camera_tokens = sum([
                    self.nusc_proc.gen_tokens(
                        is_train=False, specified_cams=camera_channels)], [])
        else:
            scenes = self.nusc_proc.get_avail_scenes(scene_names, check_all=True)
            if len(scenes) == 0:
                raise RuntimeError(
                        'No qualified scenes were found.\n'\
                        'Please check if pass_filters were properly defined, '
                        ' and if the specified scenes contained in the '\
                        'downloaded raw data.')
            self.all_camera_tokens = []
            for camera in camera_channels:
                for scene in scenes:
                    camera_tokens = self.nusc_proc.get_camera_sample_data(
                            scene, camera
                            )
                    self.all_camera_tokens.extend(camera_tokens)

        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        while self.idx >= len(self.all_camera_tokens):
            raise StopIteration

        camera_token = self.all_camera_tokens[self.idx]
        camera_sample_data = self.nusc_proc.nusc.get('sample_data',
                camera_token)

        img_path = os.path.join(self.nusc_proc.get_data_root(),
                camera_sample_data['filename'])
        img = Image.open(img_path).convert('RGB')

        img, ratio, du, dv = image_resize(img, self.height, self.width, 0, 0)

        point_cloud_uv = self.nusc_proc.get_proj_dist_sensor(camera_token,
                sensor_type=self.fused_dist_sensor)
        point_cloud_uv = self.nusc_proc.adjust_cloud_uv(point_cloud_uv, 
                self.width, self.height, ratio, du, dv)

        if self.show_bboxes:
            bboxes, cats = self.nusc_proc.gen_2d_bboxes(camera_token)
            bboxes = self.nusc_proc.adjust_2d_bboxes(bboxes,
                    self.width, self.height, ratio, du, dv)
        else:
            bboxes, cats = [], []

        self.idx += 1

        return img, point_cloud_uv, bboxes, cats
