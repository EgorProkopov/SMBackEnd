import random

from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np

import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from LookGenerator.datasets.utils import load_image


class BasicDataset(Dataset):
    """ Dataset for opening images and returning it, without targets"""

    def __init__(self, root_dir: str, dir_name: str, transform_input=None):
        """

        Args:
            root_dir: dir where a folder is
            dir_name: folder with images
            transform_input: transform for input
        """
        super().__init__()

        self.root_dir = root_dir
        self.dir_name = dir_name
        self.transform_input = transform_input

        root = os.path.join(root_dir, dir_name)

        names_of_files = os.listdir(root)

        self._files_list = [name.split('.')[0] for name in names_of_files]
        self._extensions_list = [name.split('.')[1] for name in names_of_files]

    def __getitem__(self, idx):
        to_tensor = ToTensor()

        input_ = load_image(self.root_dir,
                            self.dir_name,
                            self._files_list[idx],
                            '.' + self._extensions_list[idx])
        input_ = to_tensor(input_)

        if self.transform_input:
            input_ = self.transform_input(input_)

        return input_

    def __len__(self):
        return len(self._files_list)
