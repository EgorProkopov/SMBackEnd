import torch
from torch import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os
from typing import Tuple

from LookGenerator.datasets.utils import load_image, DirInfo

class PoseNetDataset(Dataset):
    """
        Dataset for Pose detection
    """


    def __init__(self, image_dir: str, transform=None):
        self.root = image_dir
        self.transform = transform


    def __getitem__(idx):
        return 0