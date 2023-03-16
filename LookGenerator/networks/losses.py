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
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Раскомментить, если модель на выходе имеет сигмоиду или другую аналогичную ей функцию
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        criterion = nn.BCELoss(reduction='mean')

        bce = criterion(inputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss


class FocalLossMultyClasses(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(FocalLossMultyClasses, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        batch_size, num_classes = inputs.size(0), inputs.size(1)
        inputs = inputs.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, -1)

        ce_loss = functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

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


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, smooth=1, weight=2.0):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha, gamma, smooth)
        self.dice = DiceLoss()
        self.weight = weight

    def forward(self, inputs, target):

        fcl = self.focal(inputs, target)
        dc = self.dice(inputs, target)
        fd = self.weight*fcl + dc

        return fd


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=1, smooth=1, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.weight = weight

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = functional.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky