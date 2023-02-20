import torch
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

        image = (image - min_tensor) / (self.max - self.min)
        return image
