import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional

import torchvision.models as models


class IoUMetricBin(nn.Module):
    """
    Metric for binary segmentation

    Class of Intersection over Union metric

    It is calculated as the ratio between the overlap of
    the positive instances between two sets, and their mutual combined values

    J(A, B) = |A and B| / |A or B| = |A and B| / (|A| + |B| - |A and B|)

    """
    def __init__(self):
        super(IoUMetricBin, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU


class FocalLossBin(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1):
        super(FocalLossBin, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        criterion = nn.BCELoss(reduction='mean')

        bce = criterion(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss


class FocalLossMulti(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, smooth=1, reduction='mean', device='cpu'):
        super(FocalLossMulti, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        self.device = device

        self.focal_loss_bin = FocalLossBin(
            alpha=self.alpha,
            gamma=self.gamma,
            smooth=self.smooth
        )

    def forward(self, outputs, targets):
        targets = targets.float()
        batch_size, num_labels = outputs.size(0), outputs.size(1)

        cum_loss = torch.Tensor([0]).to(self.device)

        for label in range(num_labels):
            output_channel = outputs[:][label]
            target_channel = targets[:][label]

            cum_loss += self.focal_loss_bin(output_channel, target_channel)

        if self.reduction == 'sum':
            focal_loss = cum_loss
        elif self.reduction == 'mean':
            focal_loss = cum_loss / num_labels

        return focal_loss


        # inputs = inputs.view(batch_size, num_classes, -1)
        # targets = targets.view(batch_size, -1)
        #
        # ce_loss = functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # pt = torch.exp(-ce_loss)
        # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        #
        # if self.reduction == 'mean':
        #     focal_loss = torch.mean(focal_loss)
        # elif self.reduction == 'sum':
        #     focal_loss = torch.sum(focal_loss)
        #
        # return focal_loss


class DiceLossBin(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLossBin, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + self.smooth) / (inputs.sum() + targets.sum())

        return 1 - dice


class FocalDiceLossBin(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, smooth=1, weight=2.0):
        super(FocalDiceLossBin, self).__init__()
        self.focal_bin = FocalLossBin(alpha, gamma, smooth)
        self.dice = DiceLossBin()
        self.weight = weight

    def forward(self, inputs, target):

        fcl = self.focal_bin(inputs, target)
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
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class VGG16IntermediateOutputs(nn.Module):
    # TODO: сделать возможность подгружать веса отдельно
    """
    Class to get VGG16 model intermediate outputs to calculate perceptual loss
    """
    def __init__(self, device, layer_name_mapping=None):
        super(VGG16IntermediateOutputs, self).__init__()
        vgg16_model = models.vgg16(pretrained=True)
        vgg16_model.to(device)
        vgg16_model.eval()
        self.vgg_layers = vgg16_model.features

        self.layer_name_mapping = layer_name_mapping
        if self.layer_name_mapping:
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


class PerPixelLoss(nn.Module):
    """
    L1-loss
    """
    def __init__(self):
        super(PerPixelLoss, self).__init__()

    def forward(self, outputs, targets):
        return torch.mean(torch.abs(targets - outputs))


class PerceptualLoss(nn.Module):
    """
    This perceptual loss calculate MSE loss between vgg16 activation maps
    of model output image and target image
    """
    def __init__(self, device, weights_perceptual=[1.0, 1.0, 1.0, 1.0]):
        """
        Args:
            device: device on which loss will be calculated
            weight_mse: weight for mse calculation
            weights_perceptual: weights for loss calculation of different vgg layers
        """
        super(PerceptualLoss, self).__init__()
        self.vgg16 = VGG16IntermediateOutputs(device)
        self.mse = nn.MSELoss()
        self.weights_perceptual = weights_perceptual

    def forward(self, outputs, targets):
        """
        Args:
            outputs: reconstructed image, the output of our neural net
            targets: the target image

        Returns:
            Sum of MSE losses of different vgg model outputs and L1 loss
        """
        outputs_vgg, targets_vgg = self.vgg16(outputs), self.vgg16(targets)

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

    def forward(self, discriminator, fake_image, real_image, device):
        discriminator = discriminator.to(device)
        t = torch.full(real_image.shape, np.random.rand(1)[0]).to(self.device)
        interpolation = t * real_image + (1 - t) * fake_image
        interpolation.requires_grad_()

        predicts = discriminator(interpolation)
        grads = torch.autograd.grad(
            outputs=predicts, inputs=interpolation,
            grad_outputs=torch.ones_like(predicts),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()

        return gradient_penalty


class FineGANLoss(nn.Module):
    """
    Loss for FineGAN model
    """
    def __init__(
            self, adversarial_criterion, adv_loss_weight=1,
            l1_criterion=True, l1_loss_weight=4,
            perceptual=True, perceptual_loss_weight=1,
            device=None
    ):
        """

        Args:
            adversarial_criterion: the adversarial part of GAN loss (can be BCE or MSE or something like this)
            adv_loss_weight: weight of the adversarial part of GAN loss
            l1_criterion: l1 part of loss
            l1_loss_weight: weight of lo1 part of loss
            perceptual: perceptual part of GAN loss
            perceptual_loss_weight: weight of perceptual part of loss
            device: device on which computation will be performed (it is necessary to use perceptual loss)
        """
        super(FineGANLoss, self).__init__()

        self.adversarial_criterion = adversarial_criterion
        self.adv_loss_weight = adv_loss_weight

        self.l1_criterion = None
        self.l1_loss_weight = 0
        if l1_criterion:
            self.l1_criterion = PerPixelLoss()
            self.l1_loss_weight = l1_loss_weight

        self.perceptual_criterion = None
        self.perceptual_loss_weight = 0
        if perceptual:
            self.perceptual_criterion = PerceptualLoss(device=device)
            self.perceptual_loss_weight = perceptual_loss_weight

    def forward(self, preds, targets, output_images, real_images):
        adversarial_loss = self.adv_loss_weight * self.adversarial_criterion(preds, targets)
        if self.perceptual_criterion:
            perceptual_loss = self.perceptual_loss_weight * self.perceptual_criterion(output_images, real_images)
            adversarial_loss += perceptual_loss

        if self.l1_criterion:
            l1_loss = self.l1_loss_weight * self.l1_criterion(output_images, real_images)
            adversarial_loss += l1_loss

        return adversarial_loss


class VAELoss(nn.Module):
    def __init__(self, recon_coeff=1, kld_coeff=0.5, recon_loss=nn.MSELoss()):
        """

        Args:
            recon_coeff: coefficient of reconstruction part
            kld_coeff: coefficient of kl divergence part
            recon_loss: loss to be used in reconstruction part
        """
        super(VAELoss, self).__init__()

        self.recon_coeff = recon_coeff
        self.kld_coeff = kld_coeff
        self.recon_loss = recon_loss

    def _kl_divergence(self, mu, log_var):
        """
        Args:
            mu:
            log_var:
        """

        loss = torch.mean(- 1 / 2 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # if loss > 1000.0 or loss == float('inf') or loss == float('nan'):
        #     return 1000.0
        return loss

    def _log_likelihood(self, x, reconstruction):
        """
        Args:
            x:
            reconstruction:
        """
        loss = self.recon_loss

        return loss(reconstruction, x)

    def forward(self, x, mu, log_var, reconstruction):
        kld = self._kl_divergence(mu, log_var)
        recon = self._log_likelihood(x, reconstruction)
        return self.recon_coeff * recon + self.kld_coeff * kld
