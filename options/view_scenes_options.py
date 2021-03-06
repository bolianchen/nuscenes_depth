# Copyright © 2022, Bolian Chen. Released under the MIT license.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import argparse
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ViewScenesOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # PATHS OPTIONS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 default="nuscenes_data",
                                 help="absolute or relative path to the "
                                      "root data folders") 

        self.parser.add_argument("--save_dir",
                                 type=str,
                                 default="",
                                 help="subfolder name in the project folder "
                                      "to save the rendered scene images "
                                      "leave it empty to diplay images in "
                                      "real time") 

        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        ## OPTIONS for SPECIFIC DATASET PREPROCESSING for NUSCENES DATASETS
        self.parser.add_argument("--nuscenes_version",
                            type=str,
                            default ="v1.0-mini",
                            choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
                            help="nuscenes dataset version")
        self.parser.add_argument("--camera_channels",
                            default =["CAM_FRONT"],
                            nargs="+",
                            help="selectable from CAM_FRONT, CAM_FRONT_LEFT, "
                                 "CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, "
                                 "CAM_BACK_RIGHT")
        self.parser.add_argument("--scene_names",
                                 nargs="+",
                                 type=str,
                                 default=[],
                                 help="scenes to iterate over; "
                                      "leave empty to iterate all "
                                      "available scenes of the train split")
        self.parser.add_argument("--pass_filters",
                                 nargs="+",
                                 type=str,
                                 default=['day', 'night', 'rain'],
                                 help="['day', 'night', 'rain']: all the scenes; "
                                      "['day']: daytime and not rainy scenes; "
                                      "['night']: nighttime and not rainy scenes; "
                                      "['rain']: rainy scenes on both daytime and nighttime; "
                                      "['day', 'night']: daytime, nighttime, and not rainy scenes; "
                                      "['day', 'rain']: rainy scenes on daytime; "
                                      "['night', 'rain']: rainy scenes on nighttime;")
        self.parser.add_argument("--use_keyframe",
                                 action="store_true",
                                 help="whether to use keyframes "
                                      "there are two categories: "
                                      "1. sample_data frames in 12Hz (default) "
                                      "2. keyframes in 2Hz")
        self.parser.add_argument("--show_bboxes",
                                 action="store_true",
                                 help="set true to show 2d bboxes "
                                      "set use_keyframe True to see bboxes "
                                      "on every frame; otherwise, 2d bboxes "
                                      "would not be shown on all frames "
                                      "since bbox annotations only available "
                                      "on keyframes")
        self.parser.add_argument("--show_bbox_cats",
                                 action="store_true",
                                 help="set True to display bbox categories; "
                                      "this functionality is not optimized "
                                      "and only for test")
        self.parser.add_argument("--stationary_filter",
                                 action="store_true",
                                 help="set True to filter out "
                                      "non-movable objects including "
                                      "traffic cones, barriers, "
                                      "debris and bicycle racks")
        self.parser.add_argument("--speed_bound",
                            default=[0, np.inf],
                            type=float,
                            nargs="+",
                            help="lower and upper speed limits to screen "
                                 "samples; simpy type -inf or inf "
                                 "while using command line")
        self.parser.add_argument("--fused_dist_sensor",
                                 type=str,
                                 default="radar",
                                 help="which distance sensor to be fused"
                                      "with camera image")

    def parse(self):
        self.options = self.parser.parse_args()
        self.options.project_dir = project_dir
        self.options.data_path = os.path.abspath(
                os.path.expanduser(self.options.data_path)
                )

        if self.options.save_dir:
            if not os.path.exists(self.options.save_dir):
                os.makedirs(self.options.save_dir)

        return self.options
