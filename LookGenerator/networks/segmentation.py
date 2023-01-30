import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LookGenerator.networks.modules import Conv5x5


class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=(64, 128, 256, 512)
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(Conv5x5(in_channels, feature, batch_norm=True))
            in_channels = feature

        # Decoder
        in_channels = features[-1]*2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels, feature, kernel_size=2, stride=2,
                )
            )
            in_channels = feature
            self.ups.append(Conv5x5(feature*2, feature, batch_norm=True))

        self.bottleneck = Conv5x5(features[-1], features[-1]*2, batch_norm=True)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def train_unet(model, train_dataloader, val_dataloader, device='cpu', epoch_num=5):
    model = model.to(device)

    train_history = []
    val_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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


