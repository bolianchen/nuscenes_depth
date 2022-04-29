# This module is modified from the official monodepth repository
# https://github.com/nianticlabs/monodepth2
# You may check its license to apply the codes

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms

from utils import image_resize

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg', 
                 not_do_color_aug=False,
                 not_do_flip=False,
                 do_crop=False,
                 crop_bound=[0.0, 1.0], #
                 seg_mask='none',
                 boxify=False,
                 MIN_OBJECT_AREA=20,
                 use_radar=False,
                 use_lidar=False,
                 min_depth=0.1,
                 max_depth=100.0,
                 prob_to_mask_objects=0.0, **kwargs):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        # default [0, -1, 1]
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.do_crop = do_crop
        self.crop_bound = crop_bound
        self.not_do_color_aug = not_do_color_aug
        self.not_do_flip = not_do_flip
        self.seg_mask = seg_mask
        self.boxify = boxify
        self.MIN_OBJECT_AREA = MIN_OBJECT_AREA
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.prob_to_mask_objects = prob_to_mask_objects

        # PIL image loader
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # to test if an error occur
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # check if depth data available
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same 
        augmentation to all images in this item. This ensures that all images 
        input to the pose network receive the same augmentation.
        """
        # list(inputs) is a list composed of the keys of inputs
        # radar or lidar does not need multiple scales
        for k in list(inputs):
            if "color" in k or "mask" in k:
                n, im, _ = k
                # save images resized to different scales to inputs
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # convert images to tensors
                inputs[(n, im, i)] = self.to_tensor(f)
                # augment images
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "mask" in k:
                n, im, i = k
                # convert masks to tensors
                if self.seg_mask != 'none':
                    inputs[(n, im, i)] = torch.from_numpy(np.array(f))
                else:
                    inputs[(n, im, i)] = self.to_tensor(f)
            elif "radar" in k or "lidar" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.from_numpy(np.array(f))

        if self.seg_mask != 'none':
            self.process_masks(inputs, self.seg_mask)

            if random.random() < self.prob_to_mask_objects:
                self.mask_objects(inputs)

    def process_masks(self, inputs, mask_mode):
        """
        """
        MIN_OBJECT_AREA = self.MIN_OBJECT_AREA

        for scale in range(self.num_scales):

            if mask_mode == 'color':
                object_ids = torch.unique(torch.cat(
                    [inputs['mask', fid, scale] for fid in self.frame_idxs]),
                    sorted=True)
            else:
                object_ids = torch.Tensor([0, 255])

            for fid in self.frame_idxs:
                current_mask = inputs['mask', fid, scale]

                def process_obj_mask(obj_id, mask_mode=mask_mode):
                    """Create a mask for obj_id, skipping the background mask."""
                    if mask_mode == 'color':
                        mask = torch.logical_and(
                                torch.eq(current_mask, obj_id),
                                torch.ne(current_mask, 0)
                                )
                    else:
                        mask = torch.ne(current_mask, 0)

                    # TODO early return when obj_id == 0
                    # Leave out very small masks, that are most often errors.
                    obj_size = torch.sum(mask)
                    if MIN_OBJECT_AREA != 0:
                        mask = torch.logical_and(mask, obj_size > MIN_OBJECT_AREA)
                    if not self.boxify:
                      return mask
                    # Complete the mask to its bounding box.
                    binary_obj_masks_y = torch.any(mask, axis=1, keepdim=True)
                    binary_obj_masks_x = torch.any(mask, axis=0, keepdim=True)
                    return torch.logical_and(binary_obj_masks_y, binary_obj_masks_x)

                object_mask = torch.stack(
                        list(map(process_obj_mask, object_ids))
                        )
                object_mask = torch.any(object_mask, axis=0, keepdim=True)
                inputs['mask', fid, scale] = object_mask.to(torch.float32)

    def get_image(self, image, do_flip, crop_offset=-3):
        r"""
        Resize (and crop) an image to specified height and width.
        crop_offset is an integer representing how the image will be cropped:
            -3      the image will not be cropped
            -2      the image will be center-cropped
            -1      the image will be cropped by a random offset
            >0      the image will be cropped by this offset
        """
        # If crop_offset is set to -3, crop the image since the top
        # Resize the image to (self.height, self.width).
        if crop_offset == -3:            
            image, ratio, delta_u, delta_v = image_resize(image, self.height,
                                                        self.width, 0.0, 0.0) 
        # Otherwise resize the image according to self.width, 
        # and then crop the image to self.height according to crop_offset.
        else:
            raw_w, raw_h = image.size
            resize_w = self.width
            resize_h = int(raw_h * resize_w / raw_w)
            image, ratio, delta_u, delta_v = image_resize(image, resize_h,
                                                          resize_w, 0.0, 0.0)
            top = int(self.crop_bound[0] * resize_h)
            if len(self.crop_bound) == 1:
                bottom = top
            elif len(self.crop_bound) == 2:
                bottom = int(self.crop_bound[1] * resize_h) - self.height
            else:
                raise NotImplementedError

            if crop_offset == -1:
                assert bottom >= top, "Not enough height to crop, please set a larger crop_bound range"
                # add one to include the upper limit for sampling
                crop_offset = np.random.randint(top, bottom + 1)
            elif crop_offset == -2:
                crop_offset = int((top+bottom)/2)

            image = np.array(image)
            image = image[crop_offset: crop_offset + self.height]
            image = Image.fromarray(image)
            delta_v += crop_offset

        # if the principal point is not at center,
        # flipping would affect the camera intrinsics but not accounted here
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image, ratio, delta_u, delta_v, crop_offset

    def adjust_intrinsics(self, cam_intrinsics_mat, inputs, ratio, delta_u, delta_v, do_flip):
        """Adjust intrinsics to match each scale and store to inputs"""


        for scale in range(self.num_scales):
            
            K = cam_intrinsics_mat.copy()

            # adjust K for the resizing within the get_color function
            K[0, :] *= ratio
            K[1, :] *= ratio
            K[0,2] -= delta_u
            K[1,2] -= delta_v

            # Modify the intrinsic matrix if the image is flipped
            if do_flip:
                K[0,2] = self.width - K[0,2]
            
            # adjust K for images of different scales
            K[0, :] /= (2 ** scale)
            K[1, :] /= (2 ** scale)

            inv_K = np.linalg.pinv(K)

            # add intrinsics to inputs
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

    def mask_objects(self, inputs):
        """Mask objects overlapping with mobile masks"""
        for scale in range(self.num_scales):
            for fid in self.frame_idxs:
                inputs['color_aug', fid, scale] *= (1 - inputs['mask', fid, scale])
                inputs['color', fid, scale] *= (1 - inputs['mask', fid, scale])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        # an empty dictionary
        inputs = {}

        # do augmentation?
        do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = (not self.not_do_color_aug) and do_color_aug
        # do flipping
        do_flip = self.is_train and random.random() > 0.5
        do_flip = (not self.not_do_flip) and do_flip

        line = self.filenames[index].split()

        # ex: 2011_09_26/2011_09_26_drive_0022_sync
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            # r or l
            side = line[2]
        else:
            side = None

        # add the images of the original scale to the inputs
        for i in self.frame_idxs:
            if i == "s": # this might be for stereo data
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, do_flip
                        )
            else:
                # get_color is to load the specified image
                inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index + i, side, do_flip
                        )

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            # add intrinsics to inputs
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # return a transform
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # delete the images of original scale
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_mask(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
