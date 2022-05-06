from __future__ import absolute_import, division, print_function
import os
from tqdm import tqdm
import imageio
import numpy as np
import cv2

from PIL import Image
import datasets

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def image_resize(image, target_h, target_w, shift_h, shift_w,
                 inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the raw image size

    is_pil = isinstance(image, Image.Image)

    if is_pil:
        image = np.array(image)

    (raw_h, raw_w) = image.shape[:2]

    assert raw_h >= target_h, 'must be downscaling'
    assert raw_w >= target_w, 'must be downscaling'

    if target_h/raw_h <= target_w/raw_w:
        # calculate the ratio of the width and construct the dimensions
        r = target_w / float(raw_w)
        dim = (target_w, int(raw_h * r))

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]
        
        start = int(new_h*shift_h)
        end = start + target_h
       
        assert start >=0
        assert end <= new_h

        if len(image.shape) == 3:
            image = image[start:end,:,:]
        else:
            image = image[start:end,:]

        delta_u = 0
        delta_v = start  

    else: 
        # calculate the ratio of the height and construct the dimensions
        r = target_h / float(raw_h)
        dim = (int(raw_w * r), target_h)

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]

        start = int(new_w*shift_w)
        end = start + target_w
        image = image[:,start:end,:]

        assert start >=0
        assert end <= new_w

        if len(image.shape) == 3:
            image = image[:,start:end,:]
        else:
            image = image[:,start:end]

        delta_u = start
        delta_v = 0

    if is_pil:
        image = Image.fromarray(image)

    return image, r, delta_u, delta_v

def generate_seg_masks(img_paths, threshold=0.5, seg_mask='color',
        batch_size=4, num_workers=4):
    # import the needed modules
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    # initialize a instance of the MaskRCNN model
    # the model should be discarded after the function call
    model = maskrcnn_resnet50_fpn(pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # make dataloader
    
    img_dataset = datasets.Paths2ImagesDataset(img_paths)
    img_dataloader = DataLoader(img_dataset,
                                batch_size = batch_size,
                                num_workers = num_workers)

    print("Generating segmentation masks with Mask R-CNN")
    for images, paths in tqdm(img_dataloader):
        images = list(images.to(device))
        with torch.no_grad():
            mrcnn_results = model(images)

        for i in range(len(images)):
            mrcnn_result = mrcnn_results[i]
            path = paths[i]

            # mask post-processing
            # only consider objects with predicted scores higher than threshold
            score = mrcnn_result['scores'].detach().cpu().numpy()
            valid = (score > threshold).sum()
            masks = (mrcnn_result['masks'] > threshold).squeeze(1).detach().cpu().numpy()
            labels = mrcnn_result['labels'].detach().cpu().numpy() 
            if valid > 0:
                masks = masks[:valid] # (N, H, W)
                labels = labels[:valid]
            else:
                masks = np.zeros_like(masks[:1])
                labels = np.zeros_like(labels[:1])
            masks = masks.astype(np.uint8)

            # Throw away the masks that are not pedestrians or vehicles
            masks[labels == 0] *= 0 # __background__
            masks[labels == 5] *= 0 # airplane
            masks[labels == 7] *= 0 # train
            masks[labels > 8] *= 0

            # Color ids for masks
            COLORS = np.arange(1, 256, dtype=np.uint8).reshape(-1, 1, 1)

            # TODO: self.mask
            mask_img = np.ones_like(masks, dtype=np.uint8) 
            if seg_mask == 'mono':
                mask_img = masks * mask_img
                mask_img = np.sum(mask_img, axis=0)
                mask_img = (mask_img > 0).astype(np.uint8) * 255
                return mask_img
            elif seg_mask == 'color':
                for i in range(masks.shape[0]-1):
                    masks[i+1:] *= 1 - masks[i]
                # ignore this step when masks is empty 
                if masks.shape[0] != 0:
                    # for non-background objects
                    # sample colors evenly between 1 and 255
                    mask_img = masks * mask_img * COLORS[
                            np.linspace(0, 254, num= masks.shape[0], dtype=np.uint8)
                            ]
                mask_img = np.sum(mask_img, axis=0)

            mask_path = os.path.splitext(path)[0] + '-fseg.jpg'
            imageio.imsave(mask_path, mask_img.astype(np.uint8))


