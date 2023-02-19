import torch
import torch.nn as nn
import torchvision


class RefinementGenerator(nn.Module):
    def __init__(self):
        super(RefinementGenerator, self).__init__()

    def forward(self, x):
        pass


class RefinementDiscriminator(nn.Module):
    def __init__(self):
        super(RefinementDiscriminator, self).__init__()

    def forward(self, x):
        pass


def train_refinement_network(model, train_dataloader, val_dataloader,
                             optimizer, device='cpu', epoch_num=5, save_directory=None):
    pass
