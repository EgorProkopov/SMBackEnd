import random

from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np

import os
from typing import Tuple
from multipledispatch import dispatch

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """ Dataset of images for inner tasks"""

    def __init__(self, root_dir: str, dir_name: str):
        """
        Args:
            root_dir (str) : path to images
            dir_name (str) : images directory name
        """

        super().__init__()

        self.root = os.path.join(root_dir, dir_name)

        names_of_files = os.listdir(self.root)

        self._files_list = [name.split('.')[0] for name in names_of_files]
        self._extensions_list = [name.split('.')[1] for name in names_of_files]

    @dispatch(int)
    def __getitem__(self, idx: int) -> np.array:
        """
        Opening image by index that considered self.names_of_files variable

        Args:
            idx: The index of data sample

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
        """
        Opening image by its name.

        Args:
            name: name of the file

        Returns:
            Return np.array that represent image
        """
        input_ = np.array(Image.open(os.path.join(self.root,
                                     name + "." + self._extensions_list[self.__get_files_list__().index(name)])))
        return input_

    @dispatch(int, int, int)
    def __getitem__(self, idx: int, width: int, height: int):
        """
        Opening image by index and resizing it to (width, height) size

        Args:
            name: name of the file

        Returns:
            Return np.array that represent image
        """
        return np.array(ToPILImage()(self.__getitem__(idx)).resize((width, height)))

    @dispatch(str, int, int)
    def __getitem__(self, name: str, width: int, height: int):
        """
        Opening image by its name and resizing it to (width, height) size

        Args:
            name: name of the file

        Returns:
            Return np.array that represent image
        """
        return np.array(ToPILImage()(self.__getitem__(name)).resize((width, height)))

    def __get_name__(self, idx):
        """
        Returning name of file and its extension

        Args:
            The index of data sample

        Return:
            name of file and its extension
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
    """ Dataset class for background cutting task"""
    def __init__(self, image_root_dir: str,
                 dir_name_person: str,
                 dir_name_mask: str,
                 background_root_dir: str,
                 dir_name_background: str,
                 transform_input=None,
                 transform_output=None,
                 augment=None):
        """
        Args:
            @param image_root_dir: path to the directory of images
            @param dir_name_person: the name of the image directory
            @param dir_name_mask: the name of the mask directory
                @note so this is mean that your image is located in "image_root_dir/dir_name_person"

            @param background_root_dir: path to the directory of backgrounds
            @param dir_name_background: the name of the background directory
                @note so this is mean that your background is located in "background_root_dir/dir_name_background"

            @param transform_input:
            @param transform_output:
            @param augment :
        """

        self.person_dataset = ImageDataset(image_root_dir, dir_name_person)
        self.mask_dataset = ImageDataset(image_root_dir, dir_name_mask)
        self.background = ImageDataset(background_root_dir, dir_name_background)

        self.dir_name_person = dir_name_person
        self.dir_name_mask = dir_name_mask
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.augment = augment

        self._files_list = self.mask_dataset.__get_files_list__()

    def __get_files_list__(self):
        """
        @return: array that contains names of files
        """
        return self._files_list

    def __len__(self):
        """
        @return: int that is representing length of names of files
        """
        return len(self.__get_files_list__())

    def _get_changed_background_image(self, idx, layer_to_change: int) -> Tuple[np.array, np.array]:
        """
        Returning image with changed background

        @param idx: (int)
        @param layer_to_change: (int) the number of layer that we want to replace with background
        @return: (np.array)
        """
        result = self.person_dataset.__getitem__(idx)
        mask = self.mask_dataset.__getitem__(idx)

        # get resized background
        background = self.background.__getitem__(random.randint(0, self.background.__len__() - 1),
                                                 mask.shape[1],
                                                 mask.shape[0])

        layer = (mask == layer_to_change)
        result[layer] = background[layer]

        return result, layer.astype(int)

    def __getitem__(self,  idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @param idx: (int)
        @return: image and its mask
        """
        input_, target = self._get_changed_background_image(idx, layer_to_change=0)

        to_tensor = ToTensor()

        if self.augment:
            transformed = self.augment(image=input_,
                                       mask=target)
            input_ = transformed['image']
            target = transformed['mask']
        input_ = to_tensor(input_)
        target = to_tensor(target)
        if self.transform_input:
            input_ = self.transform_input(input_)

        if self.transform_output:
            target = self.transform_output(target)

        return input_, target
