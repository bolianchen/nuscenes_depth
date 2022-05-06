# All rights reserved.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import argparse
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SimulateDataLoaderOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # PATHS OPTIONS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 default="nuscenes_data",
                                 help="absolute or relative path to the "
                                      "root data folders")

        # OPTIONS for GENERAL DATASET PREPROCESSING for ALL DATASETS
        self.parser.add_argument("--dataset",
                                 help="dataset to load",
                                 default="nuscenes",
                                 choices=["nuscenes"])
        self.parser.add_argument("--subset_ratio",
                                 type=float,
                                 default=1.0,
                                 help="random sample a subset of the dataset") 
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 default=[0, 1, 2, 3],
                                 help="scales used in the loss, this affects "
                                      "both dataloader and some methods to "
                                      "compute losses")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        ## DATA AUGMENTATION OPTIONS
        self.parser.add_argument("--not_do_color_aug",
                                 help="whether to do color augmentation",
                                 action="store_true")
        self.parser.add_argument("--not_do_flip",
                                 help="whether to flip image horizontally ",
                                 action="store_true")
        self.parser.add_argument("--do_crop",
                                 help="whether to crop image height",
                                 action="store_true")
        self.parser.add_argument("--crop_bound",
                                 type=float, nargs="+",
                                 help="for example, crop_bound=[0.0, 0.8]"
                                      " means the bottom 20% of the image will"
                                      " never be cropped. If only one value is"
                                      " given, only the top will be cropped"
                                      " according to the ratio",
                                 default=[0.0, 1.0])
        ## POSSIBLY MOBILE MASKS options
                                        
        MASK = ["none", "mono", "color"]
        self.parser.add_argument("--seg_mask",
                                 type=str,
                                 choices=MASK,
                                 default="none",
                                 help="whether to use segmetation mask")
        self.parser.add_argument("--MIN_OBJECT_AREA",
                                 type=int,
                                 default=20,
                                 help="size threshold to discard mobile masks"
                                      " set as 0 to disable the size screening"
                                 )
        self.parser.add_argument("--boxify",
                                 action="store_true",
                                 help="reshape masks to bounding boxes")

        ## REMOVE MASKED OBJECTS
        self.parser.add_argument("--prob_to_mask_objects", # mixed datasets
                                 type=float,
                                 default=0.0,
                                 help="probability to remove objects "
                                      "overlapping with mobile masks."
                                      " set 0.0 to disable, set 1.0 to"
                                      " objects with 100%")

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
        self.parser.add_argument("--use_keyframe",
                                 action="store_true",
                                 help="whether to use keyframes "
                                      "there are two categories: "
                                      "1. sample_data frames in 12Hz (default) "
                                      "2. keyframes in 2Hz")
        self.parser.add_argument("--enforce_adj_nonkeyframe",
                                 action="store_true",
                                 help="this option only effective when "
                                      "use_keyframe is set True: "
                                      "if True, adjacent non-keyframes would "
                                      " be paired with the central keyframe "
                                      "for image construction; "
                                      "if False, all adjacent and central "
                                      "frames would be keyframes")
        self.parser.add_argument("--stationary_filter",
                                 action="store_true",
                                 help="set True to filter out "
                                      "non-movable objects including "
                                      "traffic cones, barriers, "
                                      "debris and bicycle racks")
        self.parser.add_argument("--speed_limits",
                            default=[0, np.inf],
                            type=float,
                            nargs="+",
                            help="lower and upper speed limits to screen samples")
        self.parser.add_argument("--use_maskrcnn_masks",
                                 action="store_true",
                                 help="only applicable for the nuScenes dataset "
                                      "to pre-generate a mask with Mask R-CNN "
                                      "and save in disk alongside each image")
        self.parser.add_argument("--use_radar",
                                 help="if set, uses radar data for training",
                                 action="store_true")
        self.parser.add_argument("--use_lidar",
                                 help="if set, uses lidar data for training",
                                 action="store_true")
        # DATALOADER & OPTIMIZATION Options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=4)
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)

        # OPTIONS to FILTER RADAR and LIDAR GROUND-TRUTH
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)

        # TENSORBOARD LOG OPTIONS
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 default="log",
                                 help="subfolder name in the project folder "
                                      "to save the tensorboard results") 
        self.parser.add_argument("--log_steps",
                                 type=int,
                                 help="number of minibatches to log",
                                 default=10)
        
    def parse(self):
        self.options = self.parser.parse_args()
        self.options.project_dir = project_dir
        self.options.data_path = os.path.abspath(
                os.path.expanduser(self.options.data_path)
                )
        if not os.path.exists(self.options.log_dir):
            os.mkdir(self.options.log_dir)
        return self.options
