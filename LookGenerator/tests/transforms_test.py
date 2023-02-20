import numpy as np

import torch
from torchvision import transforms

from LookGenerator.datasets.transforms import MinMaxScale, ThresholdTransform


def min_max_scale_test(image_):
    transform = MinMaxScale()
    image_ = transform(image_)
    print(image_)
    return image_


def threshold_transform_test(image_):
    transform = ThresholdTransform(threshold=0.8)
    image_ = transform(image_)
    print(image_)
    return image_


if __name__ == "__main__":
    image_1 = torch.rand(5, 5) * 100
    image_2 = -torch.rand(5, 5) * 100
    image = image_1 + image_2
    print(image)

    image = min_max_scale_test(image)
    image = threshold_transform_test(image)
