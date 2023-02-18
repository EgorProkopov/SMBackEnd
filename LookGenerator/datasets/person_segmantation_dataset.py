
import torch
from torch import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
from typing import Tuple

from LookGenerator.datasets.utils import load_image, convert_channel


class PersonSegmentationDatasetMultichannel(Dataset):
    """
    Dataset for a Person Segmentation task, uses multichannel mask
    Might be deprecated soon
    """

    def __init__(self, image_dir: str, transform_input=None, transform_mask=None):
        """
        Args:
            image_dir: Directory with all images
            transform_input: A transform to be applied on input images. Default: None
            transform_mask: A transform to be applied on mask of image. Default: None
        """

        super().__init__()

        self.root = image_dir
        self.transform_input = transform_input
        self.transform_mask = transform_mask

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns: A Pair of X and y objects for segmentation
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = load_image(self.root, "image", self._files_list[idx],
                             ".jpg")
        input_ = to_tensor(input_)

        if self.transform_input:
            input_ = self.transform_input(input_)

        target = torch.empty(0)
        channel_list = os.listdir(os.path.join(
                                                self.root,
                                                "image-densepose-multichannel",
                                                self._files_list[idx]
                                                ))
        channel_files_list = [file.split('.')[0] for file in channel_list]

        for channel in channel_files_list:
            target_channel = convert_channel( load_image(self.root,
                                        os.path.join("image-densepose-multichannel", self._files_list[idx]),
                                        channel,
                                        ".png") )
            target_channel = to_tensor(target_channel)

            if self.transform_mask:
                target_channel = self.transform_mask(target_channel)

            target = torch.cat((target, target_channel), axis=0)

        return input_, target

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)


class PersonSegmentationDatasetMultichannel(Dataset):
    """Dataset for a Person Segmentation task"""

    def __init__(self, image_dir: str, transform_input=None, transform_mask=None):
        """
        Args:
            image_dir: Directory with all images
            transform_input: A transform to be applied on input images. Default: None
            transform_mask: A transform to be applied on mask of image. Default: None
        """

        super().__init__()

        self.root = image_dir
        self.transform_input = transform_input
        self.transform_mask = transform_mask

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns: A Pair of X and y objects for segmentation
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = load_image(self.root, "image", self._files_list[idx], ".jpg")
        input_ = to_tensor(input_)

        if self.transform_input:
            input_ = self.transform_input(input_)

        target = load_image(self.root, "image-densepose", self._files_list[idx], ".jpg")
        target = to_tensor(target)

        if self.transform_mask:
            target = self.transform_mask(target)

        return input_, target

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)
