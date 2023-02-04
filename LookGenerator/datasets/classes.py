from torch import random
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from typing import Tuple
from torchvision.transforms import ToTensor
from dataclasses import dataclass


@dataclass
class DirInfo:
    name: str
    extension: str


def _load_image(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    return Image.open(
        os.path.join(
            root_dir,
            dir_name,
            file_name + extension
        )
    )


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
            "denspose": DirInfo("image-densepose", ".jpg"),
            "parse-agnostic": DirInfo("image-densepose", ".jpg"),
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

        input_ = _load_image(self.root, self._dir_info["image"].name, self._files_list[idx],
                             self._dir_info["image"].extension)
        input_ = to_tensor(input_)

        target = _load_image(self.root, self._dir_info[self.type].name, self._files_list[idx],
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

        input_ = _load_image(self.root, self._dir_info["image"].name, self._files_list[idx],
                             self._dir_info["image"].extension)
        input_ = to_tensor(input_)

        target = _load_image(self.root, self._dir_info["mask"].name, self._files_list[idx],
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
