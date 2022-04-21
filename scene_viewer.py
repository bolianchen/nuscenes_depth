import os
from options import SceneViewerOptions
from datasets import NuScenesIterator, NuScenesProcessor
from matplotlib import pyplot as plt

def main(opts):
    """Render the camera images fused with the specified distance sensor
    """
    nusc_proc = NuScenesProcessor(opts.nuscenes_version, opts.data_path,
            opts.frame_ids, speed_limits=opts.speed_limits,
            cameras=opts.camera_channels, use_keyframe=opts.use_keyframe)

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
    """
    """
    nusc_iterator = NuScenesIterator(
        nusc_processor, opts.width, opts.height,
        cameras=opts.camera_channels, 
        scene_names=opts.scene_names,
        fused_dist_sensor=opts.fused_dist_sensor)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.tight_layout()
    for img, points in nusc_iterator:
        ax.cla()
        ax.set_axis_off()
        ax.imshow(img)
        ax.scatter(points[0,:], points[1,:],
                c=points[2,:], s=5)
        plt.pause(1/FPS)

def display_multi_cams(opts, nusc_processor, FPS):
    """
    """

    num_cams = len(opts.camera_channels)
    # define the canvas
    if num_cams < 4:
        num_rows = 1
        num_cols = num_cams
    elif num_cams == 4:
        num_cols = 2
        num_rows = 2
    else:
        num_cols = 3
        num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols,
            figsize=(8*num_cols, 4.5*num_rows))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    nusc_iterators = []

    for camera_channel in opts.camera_channels:
        nusc_iterators.append(
                NuScenesIterator(
                    nusc_processor, opts.width, opts.height,
                    cameras=[camera_channel], 
                    scene_names=opts.scene_names,
                    fused_dist_sensor=opts.fused_dist_sensor)
                )

    for data_pairs in zip(*nusc_iterators):
        if num_rows == 1:
            [axes[i].cla() for i in range(num_cols)]
        else:
            [axes[i][j].cla() for i in range(num_rows) for j in range(num_cols)]
        
        for idx, (img, points) in enumerate(data_pairs):
            if num_rows == 1:
                ax = axes[idx%num_cols]
            else:
                ax = axes[idx//num_cols, idx%num_cols]
            ax.set_axis_off()
            ax.imshow(img)
            ax.scatter(points[0,:], points[1,:],
                    c=points[2,:], s=5)
        plt.pause(1/FPS)

if __name__ == '__main__':
    opts = SceneViewerOptions().parse()
    opts.data_path = os.path.abspath(
            os.path.expanduser(opts.data_path)
            )
    main(opts)
