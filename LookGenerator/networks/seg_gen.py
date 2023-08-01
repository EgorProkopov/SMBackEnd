import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from LookGenerator.networks.modules import Conv3x3, Conv5x5, SelfAttentionBlock, PoolingLayer2d
from LookGenerator.networks.utils import save_model


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, instance_norm=True, activation_func=nn.LeakyReLU()):
        super(DownBlock, self).__init__()

        self.conv = Conv3x3(
            in_channels=in_channels, out_channels=out_channels,
            dropout=False, batch_norm=batch_norm, instance_norm=instance_norm,
            activation_func=activation_func,
            bias=False
        )

        self.attention = SelfAttentionBlock(num_channels=out_channels, activation_func=activation_func)
        self.pooling = PoolingLayer2d(num_channels=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.attention(out)
        out = self.pooling(out)

        return out


class SubBranchUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, instance_norm=True, activation_func=nn.LeakyReLU()):
        super(SubBranchUpBlock, self).__init__()

        self.conv = Conv3x3(
            in_channels=in_channels * 2, out_channels=out_channels,
            dropout=False, batch_norm=batch_norm, instance_norm=instance_norm,
            activation_func=activation_func,
            bias=False
        )

        self.attention = SelfAttentionBlock(num_channels=out_channels, activation_func=activation_func)
        self.transpose_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.attention(out)
        out = self.transpose_layer(out)

        return out


class MainBranchUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, instance_norm=True, activation_func=nn.LeakyReLU()):
        super(MainBranchUpBlock, self).__init__()

        self.conv = Conv3x3(
            in_channels=in_channels * 3, out_channels=out_channels,
            dropout=False, batch_norm=batch_norm, instance_norm=instance_norm,
            activation_func=activation_func,
            bias=False
        )

        self.attention = SelfAttentionBlock(num_channels=out_channels, activation_func=activation_func)
        self.transpose_layer = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.attention(out)
        out = self.transpose_layer(out)

        return out


class TwoHeadUNet(nn.Module):
    def __init__(
            self, in_channels, out_channels, clothes_channels,
            down_features=(32, 64, 128), up_features=(128, 64, 32),
            batch_norm=False, instance_norm=True, activation_func=nn.LeakyReLU()
    ):
        super(TwoHeadUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.clothes_channels = clothes_channels

        self.down_features = down_features
        self.up_features = up_features

        self.sub_branch_init_conv = Conv3x3(
            in_channels=self.clothes_channels, out_channels=self.down_features[0],
            dropout=False, batch_norm=batch_norm, instance_norm=instance_norm,
            activation_func=activation_func
        )

        self.main_branch_init_conv = Conv3x3(
            in_channels=self.in_channels, out_channels=self.down_features[0],
            dropout=False, batch_norm=batch_norm, instance_norm=instance_norm,
            activation_func=activation_func
        )

        self.sub_branch_downs = nn.ModuleList(
            DownBlock(
                in_channels=self.down_features[i], out_channels=self.down_features[i + 1],
                batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func
            ) for i in range(len(self.down_features) - 1)
        )

        self.sub_branch_ups = nn.ModuleList(
            SubBranchUpBlock(
                in_channels=self.up_features[i], out_channels=self.up_features[i + 1],
                batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func
            ) for i in range(len(self.up_features) - 1)
        )

        self.main_branch_downs = nn.ModuleList(
            DownBlock(
                in_channels=self.down_features[i]*2, out_channels=self.down_features[i + 1],
                batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func
            ) for i in range(len(self.down_features) - 1)
        )

        self.main_branch_ups = nn.ModuleList(
            MainBranchUpBlock(
                in_channels=self.up_features[i], out_channels=self.up_features[i + 1],
                batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func
            ) for i in range(len(self.up_features) - 1)
        )

        self.sub_branch_final_conv = nn.Conv2d(
            in_channels=self.up_features[-1], out_channels=clothes_channels,
            kernel_size=1, stride=1, padding=0
        )

        self.main_branch_final_conv = nn.Conv2d(
            in_channels=self.up_features[-1], out_channels=out_channels,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, clothes):
        # Sub branch
        clothes = self.sub_branch_init_conv(clothes)

        sub_branch_residual_connection = []
        for down in self.sub_branch_downs:
            clothes = down(clothes)
            sub_branch_residual_connection.append(clothes)

        for i, up in enumerate(self.sub_branch_ups):
            residual_clothes = sub_branch_residual_connection[len(self.sub_branch_ups)-1 - i]
            clothes = torch.cat((clothes, residual_clothes))
            clothes = up(clothes)
            sub_branch_residual_connection.append(clothes)

        clothes = self.sub_branch_final_conv(clothes)

        # Main branch
        x = self.main_branch_init_conv


