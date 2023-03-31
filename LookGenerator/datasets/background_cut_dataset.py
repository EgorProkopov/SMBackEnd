from torchvision.transforms import ToTensor
import torch
import numpy as np
from LookGenerator.config.config import DatasetConfig

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

        names_of_files =  os.listdir(self.root)

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
    def __getitem__(self, name: str)-> np.array:
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
