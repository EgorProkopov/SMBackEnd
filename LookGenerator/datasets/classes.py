from torch import random
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from typing import Dict, Tuple, Any
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

    def __init__(self, image_dir: str, transform=None, densepose=False, parse_agnostic=False, parse=False):
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

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

        self._dir_info = {
            "image": DirInfo("image", ".jpg")
        }
        if densepose:
            self._dir_info["denspose"] = DirInfo("image-densepose", ".jpg")
        if parse_agnostic:
            self._dir_info["parse-agnostic"] = DirInfo("image-parse-agnostic-v3.2", ".png")
        if parse:
            self._dir_info["parse"] = DirInfo("image-parse-v3", ".png")

    def __getitem__(self, idx) -> tuple[Any, ...]:
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a dict of torch.Tensor objects in order:
                "image": image of person
                "densepose": image of densepose segmentation
                "parse_agnostic": image of parse agnostic segmentation
                "parse": image of parse segmentation
        """

        seed = random.seed()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        images = []
        for key in self._dir_info.keys():
            image = _load_image(self.root, self._dir_info[key].name, self._files_list[idx], self._dir_info[key].extension)
            if self.transform:
                torch.manual_seed(seed)
                image = self.transform(image)
            images.append(image)

        #sample = {
        #    "image": images[0],
        #    "densepose": images[1],
        #    "parse_agnostic": images[2],
        #    "parse": images[3]
        #}

        return tuple(images)

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

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a dict of torch.Tensor objects:
                {
                    "image": torch.Tensor of cloth image
                    "mask": torch.Tensor of cloth mask
                }
        """

        seed = random.seed()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        images = []
        for key in self._dir_info.keys():
            image = _load_image(self.root, self._dir_info[key].name, self._files_list[idx],
                                self._dir_info[key].extension)
            image = to_tensor(image)
            if self.transform:
                torch.manual_seed(seed)
                image = self.transform(image)
            images.append(image)

        sample = {
            "image": images[0],
            "mask": images[1]
        }

        return sample

    def __len__(self):
        return len(self._files_list)
