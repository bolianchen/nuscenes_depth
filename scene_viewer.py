import os
import numpy as np
from options import SceneViewerOptions
from datasets import NuScenesIterator, NuScenesProcessor
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

def main(opts):
    """Render the camera images fused with the specified distance sensor
    """
    nusc_proc = NuScenesProcessor(opts.nuscenes_version, opts.data_path,
            opts.frame_ids, speed_limits=opts.speed_limits,
            cameras=opts.camera_channels, use_keyframe=opts.use_keyframe,
            stationary_filter=opts.stationary_filter)

    # display synchronized frames from multiple cameras
    if opts.use_keyframe and len(opts.camera_channels) > 1:
        FPS = 2 # 2Hz for keyframes
        display_multi_cams(opts, nusc_proc, FPS)
    else:
        if opts.use_keyframe:
            FPS = 2
        else:
            # 12 Hz for normal camera frames
            FPS = 12
        display_single_cam(opts, nusc_proc, FPS)

def display_single_cam(opts, nusc_processor, FPS):
    """Displays or saves images of each selected camera in the specified scene
    """
    nusc_iterator = NuScenesIterator(
        nusc_processor, opts.width, opts.height,
        cameras=opts.camera_channels, 
        scene_names=opts.scene_names,
        fused_dist_sensor=opts.fused_dist_sensor,
        show_bboxes=opts.show_bboxes, 
        )
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.tight_layout()
    for frame_id, (img, points, bboxes, cats) in enumerate(nusc_iterator):

        viz_helper(img, points, bboxes, cats, ax, opts)

        if opts.save_dir: # save images in the specified folder
            plt.savefig(
                    os.path.join(opts.save_dir, f'{frame_id:08d}.png')
                    )
        else: # display images in real time
            plt.pause(1/FPS)


def display_multi_cams(opts, nusc_processor, FPS):
    """Displays or saves concatnated images of all the selected cameras
    """

    # the number of the selected cameras
    num_cams = len(opts.camera_channels)

    # define the canvas adaptive to the number of the selected cameras
    if num_cams < 4:
        num_rows = 1
        num_cols = num_cams
    elif num_cams == 4:
        num_cols = 2
        num_rows = 2
    else:
        num_cols = 3
        num_rows = 2

    # to maintain 16:9 aspect ratio for each each image, but downscaled by 2
    col_width, row_height = 8, 4.5

    fig, axes = plt.subplots(num_rows, num_cols,
            figsize=(col_width*num_cols, row_height*num_rows))
    plt.tight_layout()

    nusc_iterators = []

    for camera_channel in opts.camera_channels:
        nusc_iterators.append(
                NuScenesIterator(
                    nusc_processor, opts.width, opts.height,
                    cameras=[camera_channel], 
                    scene_names=opts.scene_names,
                    fused_dist_sensor=opts.fused_dist_sensor,
                    show_bboxes=opts.show_bboxes)
                )

    frame_id = 0

    for data_pairs in zip(*nusc_iterators):
        if num_rows == 1:
            [axes[i].cla() for i in range(num_cols)]
        else:
            [axes[i][j].cla() for i in range(num_rows) for j in range(num_cols)]
        
        for idx, (img, points, bboxes, cats) in enumerate(data_pairs):
            if num_rows == 1:
                ax = axes[idx%num_cols]
            else:
                ax = axes[idx//num_cols, idx%num_cols]

            viz_helper(img, points, bboxes, cats, ax, opts)

        if opts.save_dir: # save images in the specified folder
            plt.savefig(
                    os.path.join(opts.save_dir, f'{frame_id:08d}.png')
                    )
        else: # display images in real time
            plt.pause(1/FPS)

        frame_id += 1

def viz_helper(img, sensor_points, bboxes, cats, ax, opts):
    ax.cla()
    ax.set_axis_off()
    ax.imshow(img)
    ax.scatter(sensor_points[0,:], sensor_points[1,:],
            c=sensor_points[2,:], s=5)
    for bbox, cat in zip(bboxes, cats):
        rect = Rectangle(
                bbox[2:], bbox[0]-bbox[2], bbox[1]-bbox[3],
                linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        if opts.show_bbox_cats:
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/2.0
            ax.annotate(cat, (cx, cy))

if __name__ == '__main__':
    opts = SceneViewerOptions().parse()
    main(opts)
