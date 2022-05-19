# Copyright © 2022, Bolian Chen. Released under the MIT license.

import os
import numpy as np
import bisect
from matplotlib import pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords, \
                                                           generate_record
from nuscenes.utils.geometry_utils import view_points
from pyquaternion.quaternion import Quaternion

from PIL import Image
from lib.algos import generate_seg_masks
from lib.utils import check_if_scene_pass

# a map to determine which radars to be projected onto each camera image plane
CAM2RADARS = {
        'CAM_FRONT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT'],
        'CAM_FRONT_LEFT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
                           'RADAR_FRONT_RIGHT'],
        'CAM_FRONT_RIGHT': ['RADAR_FRONT', 'RADAR_FRONT_LEFT',
                            'RADAR_FRONT_RIGHT'],
        'CAM_BACK_LEFT': ['RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT'],
        'CAM_BACK_RIGHT': ['RADAR_FRONT_RIGHT', 'RADAR_BACK_RIGHT'],
        'CAM_BACK': ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'],
        }

# for filtering bboxes of stationary objects
STATIONARY_CATEGORIES={'movable_object.trafficcone', 'movable_object.barrier',
                       'movable_object.debris', 'static_object.bicycle_rack'}
        
class NuScenesProcessor:
    """ Preprocessor for the nuScenes Dataset 

    Based upon the official python SDK to design specific API for
    building dataloaders for unsupervised monocular depth models
    """

    def __init__(self, version, data_root, frame_ids,
            speed_bound=[0.0, np.inf], camera_channels=['CAM_FRONT'],
            pass_filters=['day', 'night', 'rain'], use_keyframe=False,
            stationary_filter=False, seg_mask='none', how_to_gen_masks='bbox',
            maskrcnn_batch_size=4, regen_masks=False):

        self.version = version
        self.data_root = data_root
        # initialize an instance of nuScenes data
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root)
        # prepare a symbol table to map a split to its scenes
        all_splits = create_splits_scenes()
        # dictionary of the official splits
        # key: split name; value: scenes in the split
        if self.version == 'v1.0-mini':
            self.usable_splits = {
                    'train': all_splits['mini_train'],
                    'val': all_splits['mini_val']}
        elif self.version == 'v1.0-trainval':
            self.usable_splits = {
                    'train': all_splits['train'],
                    'val': all_splits['val']}
        elif self.version == 'v1.0-test':
            self.usable_splits = {'val': self.all_splits['test']}
        else:
            raise NotImplementedError

        # initialize an instance of nuScenes canbus data if needed
        if speed_bound[0] == 0.0 and np.isposinf(speed_bound[1]):
            self.screen_speed = False
        else:
            self.screen_speed = True
            self.canbus = NuScenesCanBus(dataroot=data_root)
            self.speed_bound = speed_bound

        # make sure the sequence ids are sorted in ascending order
        self.frame_ids = sorted(frame_ids)

        # each scene contains info of all the sensor channels
        self.camera_channels = camera_channels

        self.pass_filters = pass_filters
        self.use_keyframe=use_keyframe
        self.stationary_filter = stationary_filter

        # which type of segmentation masks to generate 
        self.how_to_gen_masks = how_to_gen_masks

        if seg_mask != 'none' and self.how_to_gen_masks == 'maskrcnn':
            # collect all image paths of the specified camera_channels
            # of the usable_splits
            img_paths = []
            for scene_names in self.usable_splits.values():
                # collect all the image paths
                img_paths.extend(
                        self.get_img_paths(scene_names, self.camera_channels)
                        )
            generate_seg_masks(img_paths, seg_mask=seg_mask,
                    regen_masks=regen_masks, batch_size=maskrcnn_batch_size)

    def get_avail_scenes(self, scene_names, check_all=True):
        """Return the metadata of all the available scenes contained in split

        Args:
            scene_names(list of str): a list of scene names, ex:['scene-0001']
            check_all(bool): if False, only check the files existence denoted
                            by (2) in the following.

        scene_names meet the follows are filtered out:
        (1) do not have canbus data if self.screen_speed = True
        (2) data does not exist in hard drive (use CAM_FRONT as representative)
        (3) scenes that do not meet the criteria of the specified pass_filters

        By including the design of (2), we allow the use cases when users only
        download a subset of raw files.
        Ex: when version is set as v1.0-trainval, any subsets of the set of
        all the 10 blobs.tgz files as shown:
            [v1.0-trainval01_blobs.tgz - v1.0-trainval010_blobs.tgz]
        """

        scenes = self.nusc.scene

        if check_all and self.screen_speed:
            # filter out scenes that have no canbus data
            # TODO: check if 419 should be added
            canbus_blacklist = self.canbus.can_blacklist + [419]
            canbus_blacklist = set([f'scene-{i:04d}' for i in canbus_blacklist])
            scenes = [scene for scene in scenes if scene['name']
                    not in canbus_blacklist]

        scene_names = set(scene_names)

        kept_indices = []
        for idx, scene in enumerate(scenes):
            first_sample_token = scene['first_sample_token']
            first_sample = self.nusc.get('sample', first_sample_token)
            first_filename = self.nusc.get_sample_data_path(
                    first_sample['data']['CAM_FRONT'])

            # exclude the scenes whose first sample does not exist
            if os.path.isfile(first_filename) and scene['name'] in scene_names:
                scene_description = scene['description']

                # impose scene pass_filters
                if not check_all or check_if_scene_pass(scene_description, self.pass_filters):
                    kept_indices.append(idx)

        scenes = [scenes[i] for i in kept_indices]

        return scenes

    def gen_tokens(self, is_train=True, specified_cams=[] ):
        """Generate a list of camera tokens of the corresponding split of scenes
        Args:
            is_train(bool): if true, generate tokens for the scenes in the
                            training set; otherwise, for the val set
            specified_cams(list of str): if not empty, self.camera_channels
                                         will be overwritten it

        To generate the tokens of camera sample_data frames

        """

        if is_train:
            split = self.usable_splits['train']
        else:
            split = self.usable_splits['val']

        all_tokens = []
        all_scenes = self.get_avail_scenes(split, check_all=is_train)

        if len(specified_cams) == 0:
            camera_channels = self.camera_channels
        else:
            camera_channels = specified_cams

        for camera_channel in camera_channels:
            for scene in all_scenes:
                # collect all the frames satisfying the specified requirements
                # in each scene
                camera_frames = self.get_camera_sample_data(
                        scene, camera_channel)
                all_tokens.extend(camera_frames)

        return all_tokens

    def get_adjacent_token(self, token, relative_idx):
        """Returns the sample_data token of a earlier or later frame
        Args:
            token(str): sample_data token of the central frame
            relative_idx(int): frame difference relative to the central frame
                               0 represents the central frame itself;
                               +1 represents one frame later (i.e. next frame)
                               -1 represents one frame earlier (i.e. prev frame)
                               and so on...
        """

        if relative_idx == 0:
            return token

        if relative_idx > 0:
            action = 'next'
        else:
            action = 'prev'

        gap = abs(relative_idx)

        if self.use_keyframe:
            sample_data = self.nusc.get('sample_data', token)
            sensor_token = self.nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])['sensor_token']
            sensor_channel = self.nusc.get('sensor', sensor_token)['channel']
            keyframe_token = sample_data['sample_token']
            while gap != 0:
                keyframe_token = self.nusc.get('sample', keyframe_token)[action]
                gap -= 1

            token = self.nusc.get('sample',
                    keyframe_token)['data'][sensor_channel]

        else:
            while gap != 0:
                token = self.nusc.get('sample_data', token)[action]
                gap -= 1

        return token

    def gen_2d_bboxes(self, cam_token):
        """Generates a list of all the 2d bboxes coordinates for a frame

        Labels of bboxes only available for cam_token at keyframes;
        for a non-keyframe cam_token, an empty list would be returned

        Args:
            cam_token(str): a camera keyframe token
        """
        try:
            annots = self.get_2d_bboxes(cam_token)
        except ValueError: # return an empty list for a non-keyframe
            return [], []
            
        if self.stationary_filter:
            discarded_cats = STATIONARY_CATEGORIES
        else:
            discarded_cats = set()
        # list of [x_upleft, y_upleft, x_downright, y_downright]
        coords = np.array([annot['bbox_corners'] for annot in annots
            if annot['category_name'] not in discarded_cats])
        cats = [annot['category_name'] for annot in annots
                if annot['category_name'] not in discarded_cats]
        return coords, cats

    def adjust_2d_bboxes(self, bboxes, width, height, ratio, du, dv):
        """Adjust the coordniates of the 2d bboxes according to the resizing
        Args:
            width(int): width of the output image
            height(int): height of the output image
            ratio(float): downscaling ratio
            du(int): shift of the optical axis horizontally
            dv(int): shift of the optical axis vertically
        """

        if len(bboxes) == 0:
            return []

        bboxes = bboxes * ratio
        bboxes[:,0] = np.clip(bboxes[:, 0] - du, 0, width-1)
        bboxes[:,2] = np.clip(bboxes[:, 2] - du, 0, width-1)
        bboxes[:,1] = np.clip(bboxes[:, 1] - dv, 0, height-1)
        bboxes[:,3] = np.clip(bboxes[:, 3] - dv, 0, height-1)

        return bboxes

    def get_seg_mask(self, cam_token):
        """Returns the segmentation mask corresponding to the cam_token image

        Args:
            cam_token(str): a camera sample_data token
        Returns:
            mask(PIL.Image.Image)

        There are 3 types segmentation masks:
            1). generated by a pretrained Mask R-CNN model
            2). generated by overlapping bboxes
            3). full black masks

        when this method is called, which type to generate completely 
        determined by the initialization of the NuScenesProcessor instance

        """

        camera_sample_data = self.nusc.get('sample_data', cam_token)
        img_height = camera_sample_data['height']
        img_width = camera_sample_data['width']

        # initialize the mask as black
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        if self.how_to_gen_masks == 'maskrcnn':
            img_path = os.path.join(self.data_root,
                                     camera_sample_data['filename'])
            mask_path = os.path.splitext(img_path)[0] + '-fseg.png'
            mask = Image.open(mask_path)

        elif self.how_to_gen_masks == 'bbox' and self.use_keyframe:
            # bboxes are only available for keyframes

            # round the coordinates to integers
            bboxes = np.rint(self.gen_2d_bboxes(cam_token)[0]).astype(np.int32)
            bboxes = self.adjust_2d_bboxes(bboxes, img_width, img_height, 1, 0, 0)
            # fill the white color onto the regions of the bboxes
            for x1, y1, x2, y2 in bboxes:
                mask[y1:y2,x1:x2] = 255
            mask = Image.fromarray(mask)

        else:
            # full black masks as fallback
            # no additional processing required
            mask = Image.fromarray(mask)

        return mask

    def get_img_paths(self, scene_names, camera_channels):
        """Collect the image paths of the specified cameras in specified scenes
        """
        img_paths = []
        scenes = self.get_avail_scenes(scene_names, check_all=False)
        for scene in scenes:
            sample_token = scene['first_sample_token']
            keyframe = self.nusc.get('sample', sample_token)

            for camera_channel in camera_channels:
                cam_sample_data = self.nusc.get(
                        'sample_data',
                        keyframe['data'][camera_channel])

                while True:

                    img_path = os.path.join(
                            self.data_root, cam_sample_data['filename'])
                    # add the path only if the corresponding mask does not exists
                    img_paths.append(img_path)

                    if cam_sample_data['next'] == '':
                        break
                    else:
                        # update the metadata for the sample_data frame
                        cam_sample_data = self.nusc.get(
                                'sample_data', cam_sample_data['next'])

        return img_paths

    def get_camera_sample_data(self, scene, camera, token_only=True):
        """Collects all valid sample_data of a camera in a scene

        validity check is conducted in the while loop, including:
            -- canbus speed requirement
            -- existence of adjacent frames

        Args:
            scene: metadata of a scene
            camera: name of the camera channel
            token_only(bool): collect tokens if True, all metadata if False
        """

        if self.screen_speed:
            scene_veh_speed = self.get_vehicle_speed(scene)
        else:
            scene_veh_speed = None

        sample_token = scene['first_sample_token']
        keyframe = self.nusc.get('sample', sample_token)
        all_sample_data = []

        # get first sample token for the specified camera
        # iterate all the sample_data frames or sample frames(use_keyframe True)
        # collect the metadata of the corresponding sample_data frames

        # keyframe
        if self.use_keyframe:
            while True:
                if self.check_frame_validity(keyframe, scene_veh_speed,
                        use_keyframe=self.use_keyframe):

                    sample_data = self.nusc.get('sample_data',
                            keyframe['data'][camera])

                    if token_only:
                        all_sample_data.append(sample_data['token'])
                    else:
                        all_sample_data.append(sample_data)

                if keyframe['next'] == '':
                    break
                else:
                    # update the metadata for the keyframe
                    keyframe = self.nusc.get('sample', keyframe['next'])

        # non-keyframe
        else:
            sample_data = self.nusc.get('sample_data', keyframe['data'][camera])

            while True:
                if self.check_frame_validity(sample_data, scene_veh_speed,
                        use_keyframe=self.use_keyframe):

                    if token_only:
                        all_sample_data.append(sample_data['token'])
                    else:
                        all_sample_data.append(sample_data)

                if sample_data['next'] == '':
                    break
                else:
                    # update the metadata for the sample_data frame
                    sample_data = self.nusc.get(
                            'sample_data', sample_data['next'])

        return all_sample_data

    def get_vehicle_speed(self, scene):
        """Return timestamps and vehicle speed of the specified scene

        Args:
            scene: metadata of a scene
        Returns:
            veh_speed: numpy array with shape (2, N)
                       1st row => timestamps
                       2nd row => speeds
        """
        veh_monitor = self.canbus.get_messages(scene['name'], 'vehicle_monitor')
        veh_speed_curve = np.array([(m['utime'], m['vehicle_speed'])
            for m in veh_monitor])
        return veh_speed_curve.transpose()

    def get_sensor_frames_per_keyframe(self, sample_token, sensor):
        """Returns all the sample_data frames in the specified keyframe

        the keyframe is specified by the sample_token

        Args:
            sample_token: token of the keyframe
            sensor: sensor channel
        Returns:
            sensor_frames: a list of sample_data metadata in the keyframe
        """
        # obtain the refrence keyframe
        keyframe = self.nusc.get('sample', sample_token)

        prev_keyframe_exist = keyframe['prev']

        if prev_keyframe_exist:
            keyframe = self.nusc.get('sample', keyframe['prev'])

        # collecting sample_data frames synchronized by the keyframe
        sample = self.nusc.get('sample_data', keyframe['data'][sensor])

        if not prev_keyframe_exist:
            return [sample]

        sensor_frames = []
        while sample['next']:
            sample = self.nusc.get('sample_data', sample['next'])
            if not sample['is_key_frame']:
                sensor_frames.append(sample)
            else:
                sensor_frames.append(sample)
                break

        return sensor_frames

    def get_proj_dist_sensor(self, cam_token, sensor_type='radar'):
        """Returns the projected distance sensor data onto the image plane
        Args:
            cam_token: a camera sample_data token
        Returns:
            a sensor map(str): radar or lidar
        """
        camera_sample_data = self.nusc.get('sample_data', cam_token)
        camera_channel = camera_sample_data['channel']
        img_height = camera_sample_data['height']
        img_width = camera_sample_data['width']

        # find representative frames of the radars defined by CAM2RADARS
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

        return point_cloud_uv

    def adjust_cloud_uv(self, point_cloud_uv, width, height, ratio,
            delta_u, delta_v):
        """Obtains a depth map whose shape is consistent with the resized images
        Args:
            token(str): a camera sample_data token
            width(int): width of the corresponding image
            height(int): height of the corresponding image
        Returns:
            a sensor map(from radars or lidar) has shape of (width, height)
        """

        # adjust the projected coordinates by ratio, delta_u, delta_v
        # TODO: check if in-place adjustment might induce any issues
        point_cloud_uv[:2] *= ratio
        point_cloud_uv[0] -= delta_u
        point_cloud_uv[1] -= delta_v
        point_cloud_uv = point_cloud_uv[:, np.round(point_cloud_uv[0]) > 0]
        point_cloud_uv = point_cloud_uv[:, np.round(point_cloud_uv[0]) < width]
        point_cloud_uv = point_cloud_uv[:, np.round(point_cloud_uv[1]) > 0]
        point_cloud_uv = point_cloud_uv[:, np.round(point_cloud_uv[1]) < height]

        return point_cloud_uv

    def match_dist_sensor_frames(self, camera_sample_data,
            sensor_type='radar'):
        """Returns the matched radar frames from the radar channels
        
        Find the closest radar frames in timestamp with the camera_sample_data

        Args:
            camera_sample_data(str): metadata of camera sample_data
            sensor_type(str): 'radar' or 'lidar'
        """
        # define a binary search function only in this method frame
        # search the frame whose timestamp is closest to the camera frame
        # call get_sensor_frames_per_keyframe for each radar channel
        
        sample_token = camera_sample_data['sample_token']
        sample = self.nusc.get('sample', sample_token)
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

            select = np.argmin(
                    [target_timestamp - tss[idx-1], tss[idx] - target_timestamp]
                    )
            if select == 0:
                return idx-1
            else:
                return idx

        matched_frames = []

        sample_tokens = [sample_token]
        # concat prev, current and next sample tokens
        #sample_tokens = [t for t in (sample['prev'], sample_token,
        #    sample['next']) if t != '']

        for sensor_ch in sensor_channels:
            # collect sensor frames of prev, current and next keyframes
            sensor_frames = []
            for st in sample_tokens:
                sensor_frames.extend(
                        self.get_sensor_frames_per_keyframe(
                            st, sensor_ch)
                        )
            matched_idx = match(sensor_frames, camera_timestamp)
            matched_frames.append(sensor_frames[matched_idx])
        return matched_frames

    def check_frame_validity(self, sample, scene_veh_speed,
            use_keyframe = False):
        """ Check if a sample is valid
        Args:
            sample: metadata of a sample_data or a keyframe
            scene_veh_speed: the speed curve of the scene of the sample

        A frame is valid if:
            1) whose required adjacent frames exist according to frame_ids
            2) the frame itself and its adjacent frames meet the speed
               requirements

        TODO: reduce of the complexity of this method
        """

        if use_keyframe:
            retrieval_key = 'sample'
        else:
            retrieval_key = 'sample_data'

        # check if the speed of the current frame meets the requirements
        if self.screen_speed:
            if not self.is_speed_valid(sample, scene_veh_speed):
                return False

        # check whether to bypass the validity check for the adjacent frames
        if len(self.frame_ids) <= 1:
            return True

        repr_frame_ids = [self.frame_ids[0], self.frame_ids[-1]]

        # check if the existence of the required adjacents and their frames
        for f_id in repr_frame_ids:
            sample_copy = sample
            num_tracing = abs(f_id)

            if f_id < 0:
                action = 'prev'
            elif f_id > 0:
                action = 'next'
            else:
                continue

            while num_tracing > 0:
                if sample_copy[action] == '':
                    return False

                sample_copy = self.nusc.get(retrieval_key,
                                            sample_copy[action])
                if self.screen_speed:
                    if not self.is_speed_valid(sample_copy,
                                               scene_veh_speed):
                        return False
                num_tracing -= 1

        return True

    def is_speed_valid(self, sample, scene_veh_speed):
        """Check if a sample meets the speed requirement
        Args:
            sample: metadata of a sample_data or a keyframe
            scene_veh_speed: the speed curve of the scene of the sample
        """

        # Screen out samples not meeting the speed requirement 
        actual_speed = np.interp(sample['timestamp'],
                                 scene_veh_speed[0],
                                 scene_veh_speed[1])
        low_speed = self.speed_bound[0]
        high_speed = self.speed_bound[1]
        if actual_speed < low_speed or actual_speed > high_speed:
            return False

        return True

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

    def get_cam_intrinsics(self, token):
        """Returns 3x3 camera matrix according to the given token
        Args:
            token(str): a camera sample_data token
        """
        sample_data = self.nusc.get('sample_data', token)
        camera_calibration = self.nusc.get(
                'calibrated_sensor', sample_data['calibrated_sensor_token'])
        K = np.array(camera_calibration['camera_intrinsic'])
        return np.float32(K)

    def get_2d_bboxes(self, cam_token, visibilities=['', '1', '2', '3', '4']):
        """ Get the 2D annotation records for a given `sample_data_token.

        This is refactored from a module named export_2d_annotations_as_json.py 
	from the official devkit. There are local variables in the original
        function that cannot be changed after importing it.

	Args:
            cam_token: Sample data token belonging to a camera keyframe.
            visibilities(list of str): visibility filter for 2d bboxes
                the higher the value the better the visibility
        Returns:
            List of 2D annotation record that belongs to the input `sample_data_token`
        """

        # Get the sample data and the sample corresponding to that sample data.
        sd_rec = self.nusc.get('sample_data', cam_token)

        if not sd_rec['is_key_frame']:
            raise ValueError('The 2D re-projections are available only for keyframes.')

        s_rec = self.nusc.get('sample', sd_rec['sample_token'])

        # Get the calibrated sensor and ego pose record to get the transformation matrices.
        cs_rec = self.nusc.get(
                'calibrated_sensor',
                sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

        # Get all the annotation with the specified visibilties.
        ann_recs = [self.nusc.get('sample_annotation', token)for token in s_rec['anns']]
        ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in visibilities)]

        repro_recs = []

        for ann_rec in ann_recs:
            # Augment sample_annotation with token information.
            ann_rec['sample_annotation_token'] = ann_rec['token']
            ann_rec['sample_data_token'] = cam_token

            # Get the box in global coordinates.
            box = self.nusc.get_box(ann_rec['token'])

            # Move them to the ego-pose frame.
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # Move them to the calibrated sensor frame.
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)

            # Filter out the corners that are not in front of the calibrated sensor.
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners_3d = corners_3d[:, in_front]

            # Project 3d box to 2d.
            corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords)

            # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            # Generate dictionary record to be included in the .json file.
            repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, cam_token, sd_rec['filename'])
            repro_recs.append(repro_rec)

        return repro_recs

    def get_data_root(self):
        return self.data_root

    def get_nuscenes_obj(self):
        return self.nusc

    def get_version(self):
        return self.version

