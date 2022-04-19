import os
import sys
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

from . import CAM2RADARS
from .mono_dataset import pil_loader
from utils import image_resize

# TODO:
#     replace assertions

class NuScenesIterator:
    """An iterator to iterate over the selcted scenes"""

    def __init__(self, version, data_root, frame_ids, width, height, 
            speed_limits=[0.0, np.inf], cameras=['CAM_FRONT'],
            scene_names=[], use_keyframe=False, fused_dist_sensor='radar',
            visibilities=['', '1', '2', '3', '4']):
        """Constructor of the iterator
        Args:
            scenes(list of str): the format of each entry must be 'scene-xxxx'
                                 xxxx is 4 decimal digits from 0000 to 1200
            width: target width of the output image
            height: target height of the output image
        """
        self.nusc_proc = NuScenesProcessor(version, data_root, frame_ids,
                speed_limits=speed_limits, cameras=cameras,
                use_keyframe=use_keyframe)
        self.width, self.height = width, height
        self.use_keyframe = use_keyframe
        self.fused_dist_sensor = fused_dist_sensor
        self.visibilities = visibilities

        if len(scene_names) == 0:
            self.all_camera_tokens = sum(self.nusc_proc.gen_tokens(), [])
        else:
            scenes = self.nusc_proc.get_avail_scenes(scene_names)
            self.all_camera_tokens = []
            for camera in cameras:
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
        #bboxes = self.nusc_proc.get_2d_bboxes(camera_token)
        # bad design - exposure of the data in self.nusc_proc
        camera_sample_data = self.nusc_proc.nusc.get('sample_data',
                camera_token)
        img_path = os.path.join(self.nusc_proc.data_root,
                camera_sample_data['filename'])
        img = pil_loader(img_path)

        img, ratio, du, dv = image_resize(img, self.height, self.width, 0, 0)

        point_cloud_uv = self.nusc_proc.get_proj_dist_sensor(camera_token,
                sensor_type=self.fused_dist_sensor)
        point_cloud_uv = self.nusc_proc.adjust_cloud_uv(self.width, self.height,
                point_cloud_uv, ratio, du, dv)
        self.idx += 1

        return img, point_cloud_uv
        
class NuScenesProcessor:
    """ nuScenes Dataset Preprocessor

    There two main functionalities:

    1)Prepare train_list and val_list of sample_data tokens from selected cameras
    Multi-threading may be considered

    2)Let users create iterators over the data of the desired scenes

    """

    def __init__(self, version, data_root, frame_ids,
            speed_limits=[0.0, np.inf], cameras=['CAM_FRONT'],
            use_keyframe=False):

        self.version = version
        self.data_root = data_root
        # initialize an instance of nuScenes data
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root)
        # dictionary of the official splits
        # key: split name; value: scenes in the split
        self.all_splits = create_splits_scenes()
        # initialize an instance of nuScenes canbus data if needed
        if speed_limits[0] == 0.0 and np.isposinf(speed_limits[1]):
            self.screen_speed = False
        else:
            self.screen_speed = True
            self.canbus = NuScenesCanBus(dataroot=data_root)
            self.speed_limits = speed_limits

        # make sure the sequence ids are sorted in ascending order
        self.frame_ids = sorted(frame_ids)

        # each scene contains info of all the sensor channels
        self.cameras = cameras

        self.use_keyframe=use_keyframe

    def get_avail_scenes(self, scene_names):
        """Return the metadata of all the available scenes contained in split

        Args:
            scene_names(list of str): a list of scene names, ex:['scene-0001']

        scene_names meet the follows are filtered out:
        (1) do not have canbus data if self.screen_speed = True
        (2) data does not exist in hard drive (use CAM_FRONT as representative)

        By including the design of (2), we allow the use cases when users only
        download a subset of raw files.
        Ex: when version is set as v1.0-trainval, any subsets of the set of
        all the 10 blobs.tgz files as shown:
            [v1.0-trainval01_blobs.tgz - v1.0-trainval010_blobs.tgz]

        Returns:

        """

        scenes = self.nusc.scene

        if self.screen_speed:
            # filter out scenes that have no canbus data
            # TODO: check if 419 should be added
            canbus_blacklist = self.canbus.can_blacklist + [419]
            canbus_blacklist = set([f'scene-{i:04d}' for i in canbus_blacklist])
            scenes = [scene for scene in scenes if scene['name']
                    not in canbus_blacklist]

        scene_names = set(scene_names)

        # exclude the scenes whose first sample does not exist
        kept_indices = []
        for idx, scene in enumerate(scenes):
            first_sample_token = scene['first_sample_token']
            first_sample = self.nusc.get('sample', first_sample_token)
            first_filename = self.nusc.get_sample_data_path(
                    first_sample['data']['CAM_FRONT'])
            if os.path.isfile(first_filename) and scene['name'] in scene_names:
                kept_indices.append(idx)

        scenes = [scenes[i] for i in kept_indices]

        return scenes

    def gen_tokens(self):
        """Generate a list of camera tokens according to the available splits
        """
        if self.version == 'v1.0-mini':
            split_names = ['mini_train', 'mini_val']
        elif self.version == 'v1.0-trainval':
            split_names = ['train', 'val']
        elif self.version == 'v1.0-test':
            split_names = ['test']
        else:
            raise NotImplementedError

        all_split_tokens = []

        for split_name in split_names:
            split = self.all_splits[split_name]
            all_scenes = self.get_avail_scenes(split)
            split_tokens = []

            for camera in self.cameras:
                for scene in all_scenes:
                    camera_frames = self.get_camera_sample_data(
                            scene, camera, use_keyframe=self.use_keyframe)
                    split_tokens.extend(camera_frames)

            all_split_tokens.append(split_tokens)

        return all_split_tokens

    def get_camera_sample_data(self, scene, camera, use_keyframe=False,
            token_only=True):
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

        # get first sample token for the specified camera
        if use_keyframe:
            sample = keyframe
        else:
            sample = self.nusc.get('sample_data', keyframe['data'][camera])
        sample_exist = True

        all_sample_data = []

        while sample_exist:
            if self.check_frame_validity(sample, scene_veh_speed,
                    use_keyframe=use_keyframe):

                if use_keyframe:
                    sample_data = self.nusc.get('sample_data',
                                                sample['data'][camera])
                else:
                    sample_data = sample

                if token_only:
                    all_sample_data.append(sample_data['token'])
                else:
                    all_sample_data.append(sample_data)

            sample_exist = (sample['next'] != '')

            if sample_exist:
                if use_keyframe:
                    sample = self.nusc.get('sample', sample['next'])
                else:
                    sample = self.nusc.get('sample_data', sample['next'])

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

        return point_cloud_uv

    def adjust_cloud_uv(self, width, height, point_cloud_uv, ratio,
            delta_u, delta_v):
        """Obtains a depth map whose shape is consistent with the resized images
        Args:
            token: a camera sample_data token
            width:
            height:
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

    def match_dist_sensor_frames(self, camera_sample_data, sensor_type='radar'):
        """Returns the matched radar frames from the radar channels
        Args:
            sensor_type(str): 'radar' or 'lidar'
        Returns:
            
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
        low_speed = self.speed_limits[0]
        high_speed = self.speed_limits[1]
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

    def get_2d_bboxes(self, cam_token, visibilities=['', '1', '2', '3', '4']):
        """
        Get the 2D annotation records for a given `sample_data_token`.
        :param sample_data_token: Sample data token belonging to a camera keyframe.
        :param visibilities: Visibility filter.
        :return: List of 2D annotation record that belongs to the input `sample_data_token`
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

if __name__ == '__main__':
    version = 'v1.0-mini'
    nuscenes_dir = '/home/bryanchen/Coding/Projects/ss_depth/nuscenes_data'
    #cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    #           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    FPS = 12.0
    frame_ids = [0, -1, 1]
    speed_limits = [0, 20]
    cameras = ['CAM_FRONT']
    width = 512
    height = 288
    #scene_names = ['scene-0061']
    scene_names = []
    #nusc = NuScenesProcessor(version, nuscenes_dir, [0, -1, 1],
    #        speed_limits=[0, 20], cameras=cameras)
    #train_tokens, val_tokens = nusc.gen_tokens()
    nusc_iterator = NuScenesIterator(version, nuscenes_dir, frame_ids, width,
            height, speed_limits=speed_limits, cameras=cameras, 
            scene_names=scene_names)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.tight_layout()
    for img, points in nusc_iterator:
        
        ax.cla()
        #ax.set_xlim(0, width-1)
        #ax.set_ylim(0, height-1)
        ax.set_axis_off()
        ax.imshow(img)
        ax.scatter(points[0,:], points[1,:],
                c=points[2,:], s=5)
        plt.pause(1/FPS)

