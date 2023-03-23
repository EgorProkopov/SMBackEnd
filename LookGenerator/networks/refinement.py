import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv3x3, Conv5x5


class RefinementGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RefinementGenerator, self).__init__()

        self.refinement = nn.Sequential(
            Conv5x5(
                in_channels=in_channels,
                out_channels=8,
                batch_norm=True,
                activation_func=nn.ReLU()
            ),
            Conv5x5(
                in_channels=8,
                out_channels=8,
                batch_norm=True,
                activation_func=nn.ReLU()
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.refinement(x)
        return out


class RefinementDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(RefinementDiscriminator, self).__init__()

        self.features = nn.Sequential(
            # size: 3x256x192
            Conv5x5(
                in_channels=in_channels,
                out_channels=8,
                batch_norm=True,
                activation_func=nn.ReLU(),
                res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 8x128x96

            Conv5x5(
                in_channels=8,
                out_channels=16,
                batch_norm=True,
                activation_func=nn.ReLU(),
                res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 16x64x48

            Conv5x5(
                in_channels=16,
                out_channels=32,
                batch_norm=True,
                activation_func=nn.ReLU(),
                res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 32x32x24

            Conv5x5(
                in_channels=32,
                out_channels=64,
                batch_norm=True,
                activation_func=nn.ReLU(),
                res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # output size: 64x16x12
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*16*12, out_features=1024),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


def train_refinement_network(model, train_dataloader, val_dataloader,
                             optimizer, device='cpu', epoch_num=5, save_directory=None):
    pass
