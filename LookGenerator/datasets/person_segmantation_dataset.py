
import torch
from torch import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
from typing import Tuple

from LookGenerator.datasets.utils import load_image

class PersonSegmentationDataset(Dataset):
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

        input_ = load_image(self.root, "image", self._files_list[idx],
                             ".jpg")
        input_ = to_tensor(input)

        if self.transform_input:
            input_ = self.transform_input(input)

        # TODO: написать загрузку каналов маски
        target = ... # тут загрузка background'а
        for _ in _:
            target_channel = load_image(/// ///)
            target_channel = to_tensor(target_channel)

            if self.transform_mask:
                target_channel = self.transform_mask(target_channel)

            target = torch.cat((target, target_channel), axis=1)

        return input_, target

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)

