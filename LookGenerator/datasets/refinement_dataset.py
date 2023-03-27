import os

import numpy as np
import torch
import torchvision.transforms as transforms

from typing import Tuple
from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image, convert_channel


class RefinementGANDataset(Dataset):
    """Dataset for Refinement GAN model training"""
    def __init__(
            self, restored_images_dir: str,
            real_images_dir: str,
            transform_restored_images=None,
            transform_real_images=None
    ):
        """
        Args:
            restored_images_dir: dir to images from encoder-decoder
            real_images_dir: dir to target images
            transform_restored_images: transformations for images from encoder-decoder
            transform_real_images: transformations for target images
        """
        super(RefinementGANDataset, self).__init__()

        self.restored_images_dir = restored_images_dir
        self.real_images_dir = real_images_dir

        self.transform_restored_images = transform_restored_images
        self.transform_real_images = transform_real_images

        self.list_of_restored = os.listdir(self.restored_images_dir)
        self.list_of_real = os.listdir(self.real_images_dir)

    def __len__(self):
        return len(self.list_of_real)

    def __getitem__(self, idx):
        transform_to_tensor = transforms.ToTensor()

        input_image = transform_to_tensor(load_image(
            self.restored_images_dir, "",
            self.list_of_restored[idx], ""
        ))

        target_image = transform_to_tensor(load_image(
            self.real_images_dir, "",
            self.list_of_real[idx], ""
        ))

        if self.transform_restored_images:
            input_image = self.transform_restored_images(input_image)

        if self.transform_real_images:
            target_image = self.transform_real_images(target_image)

        return input_image, target_image

