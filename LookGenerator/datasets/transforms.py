import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms


class MinMaxScale(object):
    """
    Min-Max scale transform

    data_transformed = (data - data_min) / (data_max - data_min)
    """
    def __init__(self):
        self.min = 0
        self.max = 0

    def __call__(self, image):
        self.min = image.min()
        self.max = image.max()

        min_tensor = torch.full(image.shape, self.min)
        if self.max - self.min == 0:
            raise ZeroDivisionError

        image = (image - min_tensor) / (self.max - self.min)
        return image


class ThresholdTransform(object):
    """Signum function transform"""
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold: threshold to put 1 or 0. Value can be between 0 and 1
        """
        self.threshold = threshold

    def __call__(self, image):
        """
        Args:
            image: image to be transformed
        """
        return (image >= self.threshold).float()

