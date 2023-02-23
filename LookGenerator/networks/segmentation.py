import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LookGenerator.networks.losses import IoULoss, FocalLoss
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)

        # Encoder
        for feature in features:
            self.downs.append(Conv5x5(
                in_channels, feature,
                batch_norm=True, dropout=False,
                activation_func=nn.LeakyReLU())
            )
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.ups.append(Conv5x5(
                feature*2, feature,
                batch_norm=True, dropout=False,
                activation_func=nn.ReLU())
            )

        self.bottleneck = Conv3x3(
            features[-1], features[-1]*2,
            batch_norm=True, dropout=False,
            activation_func=nn.ReLU()
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid() - откомментить, если используется самописная функция активации

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
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        out = self.final_conv(x)
        # out = self.sigmoid(out)

        return out


class UNetTrainer:
    def __init__(self, optimizer, criterion, device='cpu', save_directory=r""):
        self.optimizer = optimizer,
        self.criterion = criterion,
        device = torch.device(device)
        self.device = device
        self.save_directory = save_directory
        self.criterion.to(self.device)

        self.train_history_epochs = []
        self.val_history_epochs = []

        self.train_history_batches = []
        self.val_history_batches = []

    def train(self, train_dataloader, val_dataloader, epoch_num=5):
        pass

    def _train_epoch(self, train_dataloader):
        pass

    def _val_epoch(self, val_dataloader):
        pass

    def draw_history_plots(self, ):
        """
        Draws plots of train and validation
        """
        pass


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

    train_history = []
    val_history = []

    criterion = FocalLoss() # nn.CrossEntropyLoss()  # IoULoss
    criterion.to(device)

    for epoch in range(epoch_num):
        model = model.to(device)

        train_running_loss = 0.0
        model.train()
        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            # targets = targets.reshape(-1, targets.shape[0] * targets.shape[1] * targets.shape[2] * targets.shape[3])
            #targets = torch.reshape(targets, (targets.shape[0], -1))

            outputs = model(data)
            outputs = torch.transpose(outputs, 1, 3)
            outputs = torch.transpose(outputs, 1, 2)
            #outputs = outputs.reshape(-1, outputs.shape[0] * outputs.shape[1] * outputs.shape[2] * outputs.shape[3])
            #outputs = torch.reshape(outputs, (outputs.shape[0], -1))

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

        train_loss = train_running_loss/len(train_dataloader)
        train_history.append(train_loss)
        print(f'Epoch {epoch} of {epoch_num - 1}, train loss: {train_loss:.5f}')
        torch.cuda.empty_cache()

        val_running_loss = 0.0
        model.eval()
        for data, targets in val_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            # targets = targets.reshape(-1, targets.shape[0] * targets.shape[1] * targets.shape[2] * targets.shape[3])
            #targets = torch.reshape(targets, (targets.shape[0], -1))

            outputs = model(data)
            outputs = torch.transpose(outputs, 1, 3)
            outputs = torch.transpose(outputs, 1, 2)
            #outputs = outputs.reshape(-1, outputs.shape[0] * outputs.shape[1] * outputs.shape[2] * outputs.shape[3])
            #outputs = torch.reshape(outputs, (outputs.shape[0], -1))

            loss = criterion(outputs, targets)
            val_running_loss += loss.item()

        val_loss = val_running_loss/len(val_dataloader)
        val_history.append(val_loss)
        print(f'Epoch {epoch} of {epoch_num - 1}, val loss: {val_loss:.5f}')
        torch.cuda.empty_cache()

        save_model(model.to('cpu'), path=f"{save_directory}\\unet_epoch_{epoch}_{val_loss}.pt")

    return train_history, val_history


if __name__ == "__main__":
    model = UNet()
