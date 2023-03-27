import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision.models as models


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


class VGG16IntermediateOutputs(nn.Module):
    """
    Class to get VGG16 model intermediate outputs to calculate perceptual loss
    """
    def __init__(self, device):
        super(VGG16IntermediateOutputs, self).__init__()
        vgg16_model = models.vgg16(pretrained=True)
        vgg16_model.to(device)
        vgg16_model.eval()
        self.vgg_layers = vgg16_model.features

        self.layer_name_mapping = {
                '3': "relu1_2",
                '8': "relu2_2",
                '15': "relu3_3",
                '22': "relu4_3"
            }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x

        return output


class PerceptualLoss(nn.Module):
    """
    This perceptual loss calculate MSE loss between vgg16 activation maps
    of model output image and target image
    """
    def __init__(self, device, weight_mse=4.0, weights_perceptual=[1.0, 1.0, 1.0, 1.0]):
        """
        Args:
            device: device on which loss will be calculated
            weight_mse: weight for mse calculation
            weights_perceptual: weights for loss calculation of different vgg layers
        """
        super(PerceptualLoss, self).__init__()
        self.vgg16 = VGG16IntermediateOutputs(device)
        self.mse = nn.MSELoss()
        self.weight_mse = weight_mse
        self.weights_perceptual = weights_perceptual

    def forward(self, outputs, targets):
        """
        Args:
            outputs: reconstructed image, the output of our neural net
            targets: the target image

        Returns:
            Sum of MSE losses of different vgg model outputs
        """
        outputs_vgg, targets_vgg = self.vgg16(outputs), self.vgg16(targets)

        loss_mse = self.weight_mse * self.mse(outputs, targets)

        loss_relu1_2 = self.weights_perceptual[0] * self.mse(outputs_vgg['relu1_2'], targets_vgg['relu1_2'].detach())
        loss_relu2_2 = self.weights_perceptual[1] * self.mse(outputs_vgg['relu2_2'], targets_vgg['relu2_2'].detach())
        loss_relu3_3 = self.weights_perceptual[2] * self.mse(outputs_vgg['relu3_3'], targets_vgg['relu3_3'].detach())
        loss_relu4_3 = self.weights_perceptual[3] * self.mse(outputs_vgg['relu4_3'], targets_vgg['relu4_3'].detach())

        loss = loss_relu1_2 + loss_relu2_2 + loss_relu3_3 + loss_relu4_3
        return loss


class WassersteinLoss(nn.Module):
    """
    Wasserstein loss is a loss for discriminator and generator of generative adversarial network
    """
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, outputs, targets):
        return -torch.mean(targets * outputs)


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for Wasserstein loss
    """
    def __init__(self, discriminator, device):
        """
        Returns:
            discriminator: discriminator network for prediction on image interpolation
            device: computing device
        """
        super(GradientPenalty, self).__init__()

        self.discriminator = discriminator.to(device)
        self.device = device

    def forward(self, fake_image, real_image):
        t = torch.full(real_image.shape, np.random.rand(1)[0]).to(self.device)
        interpolation = t * real_image + (1 - t) * fake_image
        interpolation.requires_grad_()

        predicts = self.discriminator(interpolation)
        grads = torch.autograd.grad(
            outputs=predicts, inputs=interpolation,
            grad_outputs=torch.ones_like(predicts),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

        return gradient_penalty
