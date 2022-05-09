#!/bin/bash

# root folder of the nuscenes dataset
NUSCENES_DIR="./nuscenes_data"

# Single camera, Single scene, Non-keyframe
## Scene-0757-radar
python view_scenes.py --data_path $NUSCENES_DIR --camera_channels CAM_FRONT \
	              --fused_dist_sensor radar --scene_names scene-0757 \
		      --save_dir testing_results/scenes_viewing/scene-0757_front_radar_nonkey

## Scene-0103-lidar
python view_scenes.py --data_path $NUSCENES_DIR --camera_channels CAM_FRONT \
	              --fused_dist_sensor radar --scene_names scene-0103 \
		      --save_dir testing_results/scenes_viewing/scene-0103_front_radar_nonkey

# Scene-0061-radar-multi-camera-view
python view_scenes.py --data_path $NUSCENES_DIR \
                      --camera_channels CAM_FRONT_LEFT CAM_FRONT CAM_FRONT_RIGHT \
		                        CAM_BACK_LEFT CAM_BACK CAM_BACK_RIGHT \
	              --fused_dist_sensor radar --scene_names scene-0061 --use_keyframe \
		      --save_dir testing_results/scenes_viewing/scene-0061_all_radar_key 
