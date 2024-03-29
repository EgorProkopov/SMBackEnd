import torch
import numpy as np
import os

from typing import Tuple
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize
from LookGenerator.datasets.utils import load_image


class PersonSegmentationDatasetMultichannel(Dataset):
    """
    DEPRECATED

    Dataset for a Person Segmentation task, uses multichannel mask
    Might be deprecated soon
    """

    def __init__(self, image_dir: str, transform_input=None, transform_output=None, augment=None):
        """
        Args:
            image_dir: Directory with all images
            transform_input: A transform to be applied on input images. Default: None
            transform_output: A transform to be applied on mask of image. Default: None
        """

        super().__init__()

        self.root = image_dir
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.augment = augment

        list_of_files = os.listdir(image_dir + r"\image")
        self._files_list = [file.split('.')[0] for file in list_of_files]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: The index of data sample

        Returns: A Pair of X and y objects for segmentation
        """

        seed = torch.random.seed()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = load_image(self.root, "image", self._files_list[idx], ".jpg")
        target = torch.tensor([])

        channel_list = os.listdir(os.path.join(
            self.root,
            "image-parse-v3-multichannel",
            self._files_list[idx]
        ))

        channel_files_list = [file.split('.')[0] for file in channel_list]

        for channel in channel_files_list:
            part = to_tensor((load_image(self.root,
                                         os.path.join("image-parse-v3-multichannel",
                                                      self._files_list[idx]),
                                         channel,
                                         ".png")))
            if self.transform_output:
                torch.manual_seed(seed)
                part = self.transform_output(part)
            target = torch.cat((target, part), dim=0)

        input_ = to_tensor(input_)

        if self.transform_input:
            torch.manual_seed(seed)
            input_ = self.transform_input(input_)

        return input_.float(), target.float()

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)


class PersonSegmentationDatasetMultichannelV2(Dataset):
    """
    DEPRECATED

    Dataset for a Person Segmentation task, uses multichannel mask
    Might be deprecated soon
    """

    def __init__(self, image_dir: str, transform_input=None, transform_output=None, augment=None):
        """
        Args:
            image_dir: Directory with all images
            transform_input: A transform to be applied on input images. Default: None
            transform_output: A transform to be applied on mask of image. Default: None
        """

        super().__init__()

        self.root = image_dir
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.augment = augment

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

        input_ = np.array(load_image(self.root, "image", self._files_list[idx], ".jpg"))

        target = []

        channel_list = os.listdir(os.path.join(
            self.root,
            "image-parse-v3.1-multichannel",
            self._files_list[idx]
        ))

        for channel in channel_list:
            target.append(np.array(load_image(
                self.root, os.path.join("image-parse-v3.1-multichannel", self._files_list[idx]),
                channel, ""
            )))

        target = np.dstack(target)

        transformed = self.augment(image=input_, mask=target)
        transformed['mask'][:, :, 0] = (~transformed['mask'][:, :, 0]).astype(np.uint8)

        input_ = to_tensor(transformed['image'])
        target = to_tensor(transformed['mask'])

        return input_, target

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)


class PersonSegmentationDataset(Dataset):
    """Dataset for a Person Segmentation task"""

    def __init__(self, image_dir: str, transform_input=None, transform_output=None, augment=None):
        """
        Args:
            image_dir: Directory with all images
            transform_input: transform to be performed on an input, from pytorch
            transform_output: transform to be performed on an output, from pytorch
            augment: transforms from albumentations to be used on image and mask
        """

        super().__init__()

        self.root = image_dir
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.augment = augment

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

        input_ = np.array(load_image(self.root, "image", self._files_list[idx], ".jpg"))
        target = np.array(load_image(self.root, "agnostic-v3.3", self._files_list[idx], ".png"))

        if self.augment:
            transformed = self.augment(image=input_, mask=target)
            input_ = transformed['image']
            target = transformed['mask']

        input_ = to_tensor(input_)
        target = to_tensor(target)

        if self.transform_input:
            input_ = self.transform_input(input_)

        if self.transform_output:
            input_ = self.transform_output(target)

        return input_, target

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)
