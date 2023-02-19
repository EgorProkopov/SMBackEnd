import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional


class IoULoss(nn.Module):
    """
    Loss for binary segmentation

    Class of Intersection over Union, or Jaccard Loss

    It is calculated as the ratio between the overlap of
    the positive instances between two sets, and their mutual combined values

    J(A, B) = |A and B| / |A or B| = |A and B| / (|A| + |B| - |A and B|)

    """
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if model contains a sigmoid or equivalent activation layer
        # inputs = functional.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return (1 - IoU) ** 2

