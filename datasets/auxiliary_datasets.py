# Copyright © 2022, Bolian Chen. Released under the MIT license.

import torch.utils.data as data
from torchvision import transforms
from .mono_dataset import pil_loader

class Paths2ImagesDataset(data.Dataset):
    """Dataset to sample image and its path from a given list of image paths
    """
    def __init__(self, img_paths):
        super(Paths2ImagesDataset, self).__init__()
        self.img_paths = img_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = pil_loader(img_path)
        img = self.transform(img)
        return img, img_path
