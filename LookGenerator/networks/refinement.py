import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv5x5, Conv7x7


class RefinementGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RefinementGenerator, self).__init__()

        self.refinement = nn.Sequential(
            Conv7x7(
                in_channels=in_channels,
                out_channels=8,
                batch_norm=True,
                activation_func=nn.ReLU()
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.Tanh()
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
                batch_norm=False,
                activation_func=nn.ReLU(),
                res_conn=False
            ),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # size: 8x128x96

            Conv5x5(
                in_channels=8,
                out_channels=16,
                batch_norm=False,
                activation_func=nn.ReLU(),
                res_conn=False
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # size: 16x64x48

            Conv5x5(
                in_channels=16,
                out_channels=32,
                batch_norm=False,
                activation_func=nn.ReLU(),
                res_conn=False
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # size: 32x32x24

            Conv5x5(
                in_channels=32,
                out_channels=64,
                batch_norm=False,
                activation_func=nn.ReLU(),
                res_conn=False
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # size: 64x16x12

            Conv5x5(
                in_channels=64,
                out_channels=128,
                batch_norm=False,
                activation_func=nn.ReLU(),
                res_conn=False
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            # output size: 8x8x6
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=8*6*6, out_features=1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.reshape(out, (out.shape[0], 8*6*6))
        out = self.classifier(out)
        return out
