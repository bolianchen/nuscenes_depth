#!/bin/bash

# root folder of the nuscenes dataset
NUSCENES_DIR="./nuscenes_data"

# Non-keyframe
## generate masks with a pretrained Mask R-CNN model
python simulate_dataloader.py --data_path $NUSCENES_DIR \
	                      --use_radar --use_lidar \
			      --seg_mask color \
		  	      --nuscenes_version v1.0-mini \
			      --how_to_gen_masks maskrcnn \
			      --log_dir testing_results/dataloader/color_maskrcnn

## generate black masks
python simulate_dataloader.py --data_path $NUSCENES_DIR \
	                      --use_radar --use_lidar \
			      --seg_mask color \
		  	      --nuscenes_version v1.0-mini \
			      --how_to_gen_masks black \
			      --log_dir testing_results/dataloader/color_black

## generate masks by overlapping 2d bboxes ground-truth provided with the nuscenes dataset
python simulate_dataloader.py --data_path $NUSCENES_DIR \
	                      --use_radar --use_lidar \
			      --seg_mask none \
		  	      --nuscenes_version v1.0-mini \
			      --how_to_gen_masks maskrcnn \
			      --log_dir testing_results/dataloader/none_maskrcnn

# Keyframe
python simulate_dataloader.py --data_path $NUSCENES_DIR \
	                      --use_radar --use_lidar \
			      --seg_mask color \
		  	      --nuscenes_version v1.0-mini \
			      --use_keyframe \
			      --how_to_gen_masks bbox \
			      --log_dir testing_results/color_bbox
