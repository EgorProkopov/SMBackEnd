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
        inputs_channel_size = inputs.shape[1]
        targets_channels_size = targets.shape[1]

        inputs = [inputs[:, i, :, :].reshape(-1) for i in range(inputs_channel_size)]
        targets = [targets[:, i, :, :].reshape(-1) for i in range(targets_channels_size)]

        IoULosses = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            total = (input_ + target).sum()
            union = total - intersection
            IoU = (intersection + smooth) / (union + smooth)
            IoULosses.append((1 - IoU) ** 2)

        IoULoss = sum(IoULosses)/len(IoULosses)
        return IoULoss
