from torch import random
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from typing import Dict
from torchvision.transforms import ToTensor
from dataclasses import dataclass


@dataclass
class DirInfo:
    name: str
    extension: str


def load_image(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    return Image.open(
        os.path.join(
            root_dir,
            dir_name,
            file_name + extension
        )
    )


class PersonSegmentationDataset(Dataset):
    """Dataset for a Person Segmentation task"""

    def __init__(self, image_dir: str, transform=None, densepose=True, parse_agnostic=True, parse=True):
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
            densepose (bool, optional): For usage of segmentation data at image-densepose. Default: True
            parse_agnostic (bool, optional): For usage of segmentation data at image-parse-agnostic. Default: True
            parse (bool, optional): For usage of segmentation data at parse-v3. Default: True
        """

        super().__init__()
        assert densepose or parse_agnostic or parse, "At least one of flags must be set at True!"

        self.root = image_dir
        self.transform = transform

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

        dir_info = [
            DirInfo("image", ".jpg"),
            DirInfo("image-densepose", ".jpg") if densepose else None,
            DirInfo("image-parse-agnostic-v3.2", ".png") if parse_agnostic else None,
            DirInfo("image-parse-v3", ".png") if parse else None,
        ]
        self._dir_info = list(filter(None, dir_info))

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a list of torch.Tensor objects in order:
                image of person
                image of densepose segmentation
                image of parse agnostic segmentation
                image of parse segmentation
            if corresponded flags are True
        """

        seed = random.seed()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        image = load_image(self.root, self._dir_info[0].name, self._files_list[idx], self._dir_info[0].extension)
        image = to_tensor(image)

        densepose = load_image(self.root, self._dir_info[1].name, self._files_list[idx], self._dir_info[1].extension)
        densepose = to_tensor(densepose)

        parse_agnostic = load_image(self.root, self._dir_info[2].name, self._files_list[idx],
                                    self._dir_info[2].extension)
        parse_agnostic = to_tensor(parse_agnostic)

        parse = load_image(self.root, self._dir_info[3].name, self._files_list[idx], self._dir_info[3].extension)
        parse = to_tensor(parse)

        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            densepose = self.transform(densepose)

            torch.manual_seed(seed)
            parse_agnostic = self.transform(parse_agnostic)

            torch.manual_seed(seed)
            parse = self.transform(parse)

        sample = {
            "image": image,
            "densepose": densepose,
            "parse_agnostic": parse_agnostic,
            "parse": parse
        }

        return sample

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
        self.root_dir = image_dir
        self.transform = transform

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

        self._dir_info = [
            DirInfo("cloth", ".jpg"),
            DirInfo("cloth-mask", ".jpg"),
        ]

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
        to_tensor = ToTensor()

        image = load_image(self.root_dir, self._dir_info[0].name, self._files_list[idx], self._dir_info[0].extension)
        image = to_tensor(image)

        mask = load_image(self.root_dir, self._dir_info[1].name, self._files_list[idx], self._dir_info[1].extension)
        mask = to_tensor(mask)

        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            mask = self.transform(mask)

        sample = {"image": image, "mask": mask}

        return sample

    def __len__(self):
        return len(self._files_list)
