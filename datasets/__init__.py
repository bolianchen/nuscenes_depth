
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

STATIONARY_CATEGORIES={'movable_object.trafficcone', 'movable_object.barrier',
                       'movable_object.debris', 'static_object.bicycle_rack'}


from .auxiliary_datasets import Paths2ImagesDataset
from .nuscenes_dataset import NuScenesDataset
from .nuscenes_preprocessor import NuScenesProcessor, NuScenesIterator

