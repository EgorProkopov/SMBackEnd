from torch.utils.data import Dataset
import os
import torch
import torchvision
from PIL import Image


class PersonSegmentationDataset(Dataset):
    """Dataset for a Person Segmentation task"""

    def __init__(self, image_dir: str, transform=None, use_densepose=True, use_parse_agnostic=True, use_parse=True):
        """
        Args:
            image_dir (string): Directory with all images
                Directory must contain such subdirectories as:
                    - image
                    - image-densepose
                    - image-parse-agnostic-v3.2
                    - image parse-v3
                Names of files at these directories must be equal for data of one sample.

            transform (callable, optional): Optional transform to be applied on images. Default: None
            use_densepose (optional): Optional usage of segmentation data at image-densepose. Default: True
            use_parse_agnostic (optional): Optional usage of segmentation data at image-parse-agnostic-v3.2. Default: True
            use_parse (optional): Optional usage of segmentation data at parse-v3. Default: True
        """

        super().__init__()
        assert use_densepose or use_parse_agnostic or use_parse, "At least one of flags must be set at True!"
        self.root_dir = image_dir
        self.transform = transform
        self._list_of_files = os.listdir(image_dir + r"\image")
        dir_info = [
            ("image", ".jpg"),
            ("image-densepose", ".jpg") if use_densepose else None,
            ("image-parse-agnostic-v3.2", ".png") if use_parse_agnostic else None,
            ("image-parse-v3", ".png") if use_parse else None,
        ]
        self._dir_info = list(filter(None, dir_info))

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of data sample

        Returns:
            Returns a bunch of torch.Tensor objects in order:
                image of person
                image of densepose segmentation
                image of parse agnostic segmentation
                image of parse segmentation
            if corresponded flags are True
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        result = []
        for directory, extension in self._dir_info:
            image = Image.open(os.path.normpath(os.path.join(
                self.root_dir,
                directory,
                self._list_of_files[idx][:-4] + extension
            )))
            image = torchvision.transforms.ToTensor()(image)
            if self.transform:
                image = self.transform(image)
            result.append(image)

        return result

    def __len__(self):
        return len(self._list_of_files)
