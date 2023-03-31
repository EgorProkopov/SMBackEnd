import random

from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np
from LookGenerator.config.config import DatasetConfig, Config

import os
from typing import Tuple
from multipledispatch import dispatch

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """ Dataset of images """

    def __init__(self, root_dir: str, dir_name: str):
        """
        Args:
            img_dir (string) : path to images
        """

        super().__init__()

        self.root = os.path.join(root_dir, dir_name)

        names_of_files = os.listdir(self.root)

        self._files_list = [name.split('.')[0] for name in names_of_files]
        self._extensions_list = [name.split('.')[1] for name in names_of_files]

    @dispatch(int)
    def __getitem__(self, idx: int) -> np.array:
        """
        Args: idx: The index of data sample
        Returns:
            Return np.array that represent image
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_ = np.array(Image.open(os.path.join(self.root,
                                     self._files_list[idx] + "." + self._extensions_list[idx])))
        return input_

    @dispatch(str)
    def __getitem__(self, name: str) -> np.array:
        input_ = np.array(Image.open(os.path.join(self.root,
                                     name + "." + self._extensions_list[self.__get_files_list__().index(name)])))
        return input_

    @dispatch(str, int, int)
    def __getitem__(self, name: str, width: int, height: int):
        return np.array(ToPILImage()(self.__getitem__(name)).resize((width, height)))

    @dispatch(int, int, int)
    def __getitem__(self, idx: int, width: int, height: int):
        return np.array(ToPILImage()(self.__getitem__(idx)).resize((width, height)))

    def __get_name__(self, idx):
        """
        Args: The index of data sample
        Return: name of file and its extension
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name_ = self._files_list[idx]
        extension_ = self._extensions_list[idx]

        return name_, extension_

    def __get_files_list__(self):
        return self._files_list

    def __len__(self):
        return len(self._files_list)


class PersonDataset(Dataset):
    def __init__(self, image_root_dir: str,
                 dir_name_person: str,
                 dir_name_mask: str,
                 background_root_dir: str,
                 dir_name_background: str,
                 transform_input=None,
                 transform_output=None,
                 augment=None):

        self.person_dataset = ImageDataset(image_root_dir, dir_name_person)
        self.mask_dataset = ImageDataset(image_root_dir, dir_name_mask)
        self.background = ImageDataset(background_root_dir, dir_name_background)

        self.dir_name_person = dir_name_person
        self.dir_name_mask = dir_name_mask
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.augment = augment

        self._files_list = self.mask_dataset.__get_files_list__()
        # if not all mask files have their own image
        # [file for file in self.mask_dataset.__get_files_list__() if file in self.person_dataset.__get_files_list__()]

    def __get_files_list__(self):
        return self._files_list

    def __len__(self):
        return len(self.__get_files_list__())

    def _get_changed_background_image(self, idx, layer_to_change: int):
        result = self.person_dataset.__getitem__(idx)
        mask = self.mask_dataset.__getitem__(idx)

        # get resized background
        background = self.background.__getitem__(random.randint(0, self.background.__len__() - 1),
                                                 mask.shape[1],
                                                 mask.shape[0])

        layer = (mask == layer_to_change)
        result[layer] = background[layer]

        return result, layer

    def __getitem__(self,  idx) -> Tuple[torch.Tensor, torch.Tensor]:
        input_, target = self._get_changed_background_image(idx, layer_to_change=0)

        to_tensor = ToTensor()

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
