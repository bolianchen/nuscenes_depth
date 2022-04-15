import os
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes

class NuScenesProcessor:
    """ nuScenes Dataset

    Prepare train_list and val_list of sample_data tokens from selected cameras
    Multi-threading may be considered

    """

    def __init__(self, version, data_root, frame_ids,
            speed_limits=[0.0, np.inf], cameras=['CAM_FRONT']):

        self.version = version
        self.data_root = data_root
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root)

        self.frame_ids = sorted(frame_ids)

        # whether to use canbus info
        if speed_limits[0] == 0.0 and np.isposinf(speed_limits[1]):
            self.use_canbus = False
        else:
            self.canbus = NuScenesCanBus(dataroot=data_root)
            self.speed_limits = speed_limits
            self.use_canbus = True

        # each scene contains info of all the sensor channels
        self.cameras = cameras

    def gen_tokens(self):
        if self.version == 'v1.0-mini':
            split_names = ['mini_train', 'mini_val']
        elif self.version == 'v1.0-trainval':
            split_names = ['train', 'val']
        elif self.version == 'v1.0-test':
            split_names = ['test']
        else:
            raise NotImplementedError

        # dictionary of the official splits
        all_splits = create_splits_scenes()

        all_split_tokens = []

        for split_name in split_names:
            split = all_splits[split_name]
            all_scenes = self.get_avail_scenes(split)
            split_tokens = []
            for camera in self.cameras:
                for scene in all_scenes:
                    camera_frames = self.get_camera_sample_data(
                            scene, camera)
                    split_tokens.extend(camera_frames)
            all_split_tokens.append(split_tokens)

        return all_split_tokens

    def get_avail_scenes(self, split):
        """Return the metadata of all the available scenes contained in split

        scenes meet the follows are filtered out:
        (1) do not have canbus data if self.use_canbus = True
        (2) data does not exist in hard drive (use CAM_FRONT as representative)
        """

        scenes = self.nusc.scene

        if self.use_canbus:
            # filter out scenes that have no canbus data
            # TODO: check if 419 should be added
            # total number = 22
            canbus_blacklist = self.canbus.can_blacklist + [419]
            canbus_blacklist = set([f'scene-{i:04d}' for i in canbus_blacklist])
            scenes = [scene for scene in scenes if scene['name']
                    not in canbus_blacklist]

        split = set(split)

        # exclude the scenes whose first sample does not exist
        # assume users may download only a subset of data
        # for example, only v1.0-trainval01_blobs.tgz is downloaded
        # out of the 10 blobs in the whole nuscenes dataset
        kept_indices = []
        for idx, scene in enumerate(scenes):
            first_sample_token = scene['first_sample_token']
            first_sample = self.nusc.get('sample', first_sample_token)
            first_filename = self.nusc.get_sample_data_path(
                    first_sample['data']['CAM_FRONT'])
            if os.path.isfile(first_filename) and scene['name'] in split:
                kept_indices.append(idx)

        scenes = [scenes[i] for i in kept_indices]

        return scenes

    def get_vehicle_speed(self, scene):
        """Return timestamps and vehicle speed of the specified scene

        output:
            veh_speed: numpy array with shape (2, N)
                       1st row => timestamps
                       2nd row => speeds
        """

        veh_monitor = self.canbus.get_messages(scene['name'], 'vehicle_monitor')
        veh_speed_curve = np.array([(m['utime'], m['vehicle_speed'])
            for m in veh_monitor])
        return veh_speed_curve.transpose()

    def get_frames_belong_keyframe(self, sample_token, sensor):
        """Collect sample_data frames of a specifide sensor of a keyframe
        Args:
            sample_token: token of the keyframe
            sensor: sensor channel
        """
        keyframe = self.nusc('sample', sample_token)

        sensor_sample = self.nusc.get(
                'sample_data',
                keyframe['data'][sensor]
                )
        sensor_samples = []
        keyframe = [sensor_sample]

        while sensor_sample['next']:
            sensor_sample = self.nusc.get(
                    'sample_data', sensor_sample['next'])
            is_key_frame = sensor_sample['is_key_frame']

            if is_key_frame:
                sensor_samples.append(keyframe)
                keyframe = [sensor_sample]
            else:
                keyframe.append(sensor_sample)

        return sensor_samples

    def get_camera_sample_data(self, scene, camera, token_only=True):
        """ Collect all valid sample_data of a camera in a scene

        validity check is conducted in the while loop, including:
            -- canbus speed requirement
            -- existence of adjacent frames

        Args:
            scene: metadata of a scene
            camera: name of the camera channel
        """

        if self.use_canbus:
            scene_veh_speed = self.get_vehicle_speed(scene)
        else:
            scene_veh_speed = None

        sample_token = scene['first_sample_token']
        keyframe = self.nusc.get('sample', sample_token)

        # get first sample_data token for the specified camera
        sample_data = self.nusc.get('sample_data', keyframe['data'][camera])
        sample_exist = True

        all_sample_data = []
        while sample_exist:
            if self.check_frame_validity(sample_data, scene_veh_speed):
                if token_only:
                    all_sample_data.append(sample_data['token'])
                else:
                    all_sample_data.append(sample_data)

            sample_exist = (sample_data['next'] != '')
            if sample_exist:
                sample_data = self.nusc.get('sample_data', sample_data['next'])

        return all_sample_data

    def check_frame_validity(self, sample_data, scene_veh_speed):
        """ Check if sample_data is valid
        Args:
            sample_data: metadata of a sample_data
            scene_veh_speed: the speed curve of the scene of the sample_data
        """

        if not self.is_speed_valid(sample_data, scene_veh_speed):
            return False

        repr_frame_ids = [self.frame_ids[0], self.frame_ids[-1]]

        for f_id in repr_frame_ids:
            sample_data_copy = sample_data
            num_tracing = abs(f_id)
            if f_id < 0:
                action = 'prev'
            elif f_id > 0:
                action = 'next'
            else:
                continue

            while num_tracing > 0:
                if sample_data_copy[action] == '':
                    return False
                sample_data_copy = self.nusc.get('sample_data',
                                                 sample_data_copy[action])
                if not self.is_speed_valid(sample_data_copy, scene_veh_speed):
                    return False
                num_tracing -= 1

        return True

    def is_speed_valid(self, sample_data, scene_veh_speed):
        """Check if the sample_data meets the speed requirement
        Args:
            sample_data: metadata of a sample_data
            scene_veh_speed: the speed curve of the scene of the sample_data
        """

        # Screen out samples not meeting the speed requirement
        if self.use_canbus:
            actual_speed = np.interp(sample_data['timestamp'],
                                     scene_veh_speed[0],
                                     scene_veh_speed[1])
            low_speed = self.speed_limits[0]
            high_speed = self.speed_limits[1]
            if actual_speed < low_speed or actual_speed > high_speed:
                return False

        return True


if __name__ == '__main__':
    version = 'v1.0-mini'
    nuscenes_dir = '/home/bryanchen/Coding/Projects/ss_depth/nuscenes_data'
    #cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    #           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    cameras = ['CAM_FRONT']
    nusc = NuScenesProcessor(version, nuscenes_dir, [0, -1, 1],
            speed_limits=[0, 20], cameras=cameras)
    train_tokens, val_tokens = nusc.gen_tokens()
