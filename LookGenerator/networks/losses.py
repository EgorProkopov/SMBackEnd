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
        # inputs = torch.sigmoid(inputs)
        # inputs_channel_size = inputs.shape[1]
        # targets_channels_size = targets.shape[1]

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.smooth=smooth

    def forward(self, inputs, targets):
        # Раскомментить, если модель на выходе имеет сигмоиду или другую аналогичную ей функцию
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        everyone = functional.binary_cross_entropy(inputs, targets, reduction='mean')
        all_exp = torch.exp(-everyone)
        focal_loss = self.alpha * (1 - all_exp) ** self.gamma * everyone

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + self.smooth) / (inputs.sum() + targets.sum())

        return 1 - dice
