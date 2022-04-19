import os
from options import SceneViewerOptions
from datasets import NuScenesIterator
from matplotlib import pyplot as plt

def main(opts):
    """Render the fused sensor data for the selected scenes
    """
    nusc_iterator = NuScenesIterator(
            opts.nuscenes_version, opts.data_path, opts.frame_ids,
            opts.width, opts.height, speed_limits=opts.speed_limits,
            cameras=opts.camera_channels, 
            scene_names=opts.scene_names)

    # the applied frames per second of nuscenes cameras, a constant
    FPS = 12.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    plt.tight_layout()
    for img, points in nusc_iterator:
        ax.cla()
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
