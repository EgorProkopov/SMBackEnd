import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LookGenerator.networks.modules import Conv3x3, Conv5x5
from LookGenerator.networks.utils import save_model


class UNet(nn.Module):
    """
    UNet model for segmentation with changeable number of layers
    """
    def __init__(
            self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)
    ):
        """

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels in the out mask
            features: tuple of layers activation maps numbers
        """
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Encoder
        for feature in features:
            self.downs.append(Conv5x5(in_channels, feature, batch_norm=True, activation_func=nn.ReLU()))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.ups.append(Conv5x5(feature*2, feature, batch_norm=True, activation_func=nn.ReLU()))

        self.bottleneck = Conv3x3(features[-1], features[-1]*2, batch_norm=True, activation_func=nn.ReLU())
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x, indices = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)


def train_unet(model, train_dataloader, val_dataloader, optimizer, device='cpu', epoch_num=5, save_directory=""):
    """
    Function for training and validation segmentation model
    Args:
        model: segmentation model for training
        train_dataloader: dataloader of train dataset
        val_dataloader: dataloader of val dataset
        optimizer: optimizer of the model
        device: device on which calculations will be performed
        epoch_num: number of training epochs

    Returns:

    """
    device = torch.device(device)
    model = model.to(device)

    train_history = []
    val_history = []

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        train_running_loss = 0.0
        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
    
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        train_loss = train_running_loss/len(train_dataloader)
        train_history.append(train_history)
        print(f'Epoch {epoch} of {epoch_num}, train loss: {train_loss:.3f}')

        val_running_loss = 0.0
        for data, targets in val_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            val_running_loss = loss.item()

        val_loss = val_running_loss/len(val_dataloader)
        val_history.append(val_loss)
        print(f'Epoch {epoch} of {epoch_num}, val loss: {val_loss:.3f}')

        save_model(model.to('cpu'), path=f"{save_directory}\\unet_epoch_{epoch_num}_{val_loss}.pt")

    return train_history, val_history


if __name__ == "__main__":
    model = UNet()
