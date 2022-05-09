import os
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from options import SimulateDataLoaderOptions
from lib.utils import normalize_image
from lib.img_processing import NuScenesProcessor
from datasets import NuScenesDataset

def main(opts):
    """Store a batch of dataloader output to tensorboard
    """
    # initialize a nuscenes preprocessor
    nusc_proc = NuScenesProcessor(opts.nuscenes_version, opts.data_path,
            opts.frame_ids, speed_limits=opts.speed_limits,
            camera_channels=opts.camera_channels,
            use_keyframe=opts.use_keyframe,
            stationary_filter=opts.stationary_filter,
            how_to_gen_masks=opts.how_to_gen_masks,
            regen_masks=opts.regen_masks, seg_mask=opts.seg_mask)

    # initialize training dataset
    dataset = NuScenesDataset(
            opts.data_path, nusc_proc, opts.height, opts.width,
            opts.frame_ids, len(opts.scales), is_train=True,
            not_do_color_aug=opts.not_do_color_aug,
            not_do_flip=opts.not_do_flip, do_crop=opts.do_crop,
            crop_bound=opts.crop_bound, seg_mask=opts.seg_mask,
            MIN_OBJECT_AREA=opts.MIN_OBJECT_AREA, boxify=opts.boxify,
            prob_to_mask_objects=opts.prob_to_mask_objects,
            use_radar=opts.use_radar, use_lidar=opts.use_lidar,
            min_depth=opts.min_depth, max_depth=opts.max_depth)

    dataloader = DataLoader(dataset, opts.batch_size, shuffle = True,
            num_workers=opts.num_workers, pin_memory=True, drop_last=True)

    for step, batch in enumerate(dataloader):
        print(f'saving {step}/{opts.log_steps} batch to tensorboard')

        log(batch, opts, step)

        if step >= opts.log_steps:
            break

def log(batch, opts, step):
    """Log information of the batch into tensorboard"""
    writer = SummaryWriter(f'./{opts.log_dir}')
    s = 0 # only use the scale 0

    for j in range(opts.batch_size):  # write a maxmimum of four images
        imgs = []

        if opts.seg_mask != 'none':
            masks = []
        if opts.use_radar:
            radars = []
        if opts.use_lidar:
            lidars = []

        for frame_id in sorted(opts.frame_ids):
            imgs.append(batch[('color', frame_id, s)][j].data)
            if opts.seg_mask != 'none':
                masks.append(batch[('mask', frame_id, s)][j].data)

            if opts.use_radar:
                radars.append(
                        normalize_image(
                            batch[('radar', frame_id, s)][j].unsqueeze(0))
                    )
            if opts.use_lidar:
                lidars.append(
                        normalize_image(
                            batch[('lidar', frame_id, s)][j].unsqueeze(0))
                    )

        writer.add_image(
            f'sensor_channels/id_{j}_image', torch.cat(imgs, axis=2), step)

        if opts.seg_mask != 'none':
            writer.add_image(
                    f'sensor_channels/id_{j}_segmask',
                    torch.cat(masks, axis=2), step)
        if opts.use_radar:
            writer.add_image(
                f'sensor_channels/id_{j}_radar', torch.cat(radars, axis=2), step)
        if opts.use_lidar:
            writer.add_image(
                f'sensor_channels/id_{j}_lidar', torch.cat(lidars, axis=2), step)

    writer.close()


if __name__ == '__main__':
    opts = SimulateDataLoaderOptions().parse()
    main(opts)
    

