import torch
import os

from torch import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Tuple
from LookGenerator.datasets.utils import load_image, DirInfo


class ClothesSegmentationDataset(Dataset):
    """Dataset for a Clothes Segmentation task"""

    def __init__(self, image_dir: str, transform=None):
        """
        Args:
            image_dir (string): Directory with all images
                Directory must contain such subdirectories as:
                    - cloth
                    - cloth-mask
                Names of files at these directories must be equal for data of one sample.

            transform (callable, optional): Optional transform to be applied on images. Default: None
        """

        super().__init__()
        self.root = image_dir
        self.transform = transform

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

        self._dir_info = {
            "image": DirInfo("cloth", ".jpg"),
            "mask": DirInfo("cloth-mask", ".jpg"),
        }

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a tuple of torch.Tensor objects input and target where
                input_: torch.Tensor of cloth image
                target: torch.Tensor of cloth mask
        """

        seed = random.seed()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = load_image(self.root, self._dir_info["image"].name, self._files_list[idx],
                            self._dir_info["image"].extension)
        input_ = to_tensor(input_)

        target = load_image(self.root, self._dir_info["mask"].name, self._files_list[idx],
                            self._dir_info["mask"].extension)
        target = to_tensor(target)

        if self.transform:
            torch.manual_seed(seed)
            input_ = self.transform(input_)

            torch.manual_seed(seed)
            target = self.transform(target)

        return input_, target

    def __len__(self):
        return len(self._files_list)
