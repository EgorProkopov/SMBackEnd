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
        # io_u_losses = []
        # for input_, target in zip(inputs, targets):
        #     intersection = (input_ * target).sum()
        #     total = (input_ + target).sum()
        #     union = total - intersection
        #     io_u = (intersection + smooth) / (union + smooth)
        #     io_u_losses.append((1 - io_u) ** 2)
        #
        # io_u_loss = sum(io_u_losses)/len(io_u_losses)
        # return io_u_loss


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        everyone = functional.binary_cross_entropy(inputs, targets, reduction='mean')
        all_exp = torch.exp(-everyone)
        focal_loss = alpha * (1 - all_exp) ** gamma * everyone

        return focal_loss
