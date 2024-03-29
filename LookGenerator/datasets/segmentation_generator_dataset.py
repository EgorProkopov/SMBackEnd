import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from LookGenerator.datasets.utils import load_image


class SegGenDataset(Dataset):
    """
    Dataset for a segmentation generation task, uses multichannel mask
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

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of data sample

        Returns: A Pair of X and y objects for segmentation
        """

        seed = torch.random.seed()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        to_tensor = ToTensor()

        input_ = np.array(load_image(self.root, "cloth", self._files_list[idx], ".jpg"))

        target = []
        target_ = []

        channel_list = os.listdir(os.path.join(
            self.root,
            "image-parse-v3-multichannel",
            self._files_list[idx]
        ))

        for channel in channel_list[1:]:
            if channel.split('_')[1] not in ['001.png', '002.png', '006.png']:
                target.append(np.array(load_image(
                    self.root, os.path.join("image-parse-v3-multichannel", self._files_list[idx]),
                    channel, ""
                )))
            else:
                target_.append(np.array(load_image(
                    self.root, os.path.join("image-parse-v3-multichannel", self._files_list[idx]),
                    channel, ""
                )))

        target = np.dstack(target)
        target_ = np.dstack(target_)

        target = np.concatenate((target, target_), axis=2)

        transformed = self.augment(image=input_, mask=target)

        input_ = to_tensor(transformed['image'])
        target = to_tensor(transformed['mask'])

        target_ = target[8:]

        input_ = torch.cat((input_, target[:8]), dim=0)

        return input_, target_

    def __len__(self):
        """
        Returns: the length of the dataset
        """

        return len(self._files_list)
