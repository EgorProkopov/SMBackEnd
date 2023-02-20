import numpy as np

import torch
from torchvision import transforms

from LookGenerator.datasets.transforms import MinMaxScale


def min_max_scale_test(image):
    transform = MinMaxScale()
    image = transform(image)
    print(image)


if __name__ == "__main__":
    image_1 = torch.rand(5, 5) * 100
    image_2 = -torch.rand(5, 5) * 100
    image = image_1 + image_2
    print(image)

    min_max_scale_test(image)
