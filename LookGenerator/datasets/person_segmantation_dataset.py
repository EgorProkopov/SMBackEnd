
import torch
from torch import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
from typing import Tuple

from LookGenerator.datasets.utils import load_image, DirInfo

class PersonSegmentationDataset(Dataset):
    """Dataset for a Person Segmentation task"""

    def __init__(self, image_dir: str, transform=None, segmentation_type="parse"):
        """
        Parameters:
            image_dir (str): Directory with all images
                Directory must contain such subdirectories as:
                    - image
                    - image-densepose
                    - image-parse-agnostic-v3.2
                    - image parse-v3
                Names of files at these directories must be equal for data of one sample.

            transform (callable, optional): A transform to be applied on images. Default: None
        """

        super().__init__()

        self.root = image_dir
        self.transform = transform
        self.type = segmentation_type

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

        self._dir_info = {
            "image": DirInfo("image", ".jpg"),
            "densepose": DirInfo("image-densepose", ".jpg"),
            "parse-agnostic": DirInfo("image-parse-agnostic-v3.2", ".png"),
            "parse": DirInfo("image-parse-v3", ".png")
        }

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a tuple of torch.Tensor input and target with type:
                "image": image of person
                "densepose": image of densepose segmentation
                "parse_agnostic": image of parse agnostic segmentation
                "parse": image of parse segmentation
        """

        seed = random.seed()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = load_image(self.root, self._dir_info["image"].name, self._files_list[idx],
                             self._dir_info["image"].extension)
        input_ = to_tensor(input_)

        target = load_image(self.root, self._dir_info[self.type].name, self._files_list[idx],
                             self._dir_info[self.type].extension)
        target = to_tensor(target)

        if self.transform:
            torch.manual_seed(seed)
            input_ = self.transform(input_)

            torch.manual_seed(seed)
            target = self.transform(target)

        return input_, target

    def __len__(self):
        return len(self._files_list)

