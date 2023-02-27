import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from LookGenerator.networks.losses import FocalLoss
from LookGenerator.networks.modules import Conv3x3, Conv5x5
from LookGenerator.networks.utils import save_model, _get_num_digits


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
        self.classifier = nn.Sequential(
            Conv5x5(features[0], features[0], batch_norm=True, dropout=False, activation_func=nn.ReLU()),
            nn.Conv2d(features[0], out_channels, kernel_size=1)
        )
        # self.sigmoid = nn.Sigmoid() #  - откомментить, если используется самописная функция активации

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

            # if x.shape != skip_connection.shape:
            #     x = transforms.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        out = self.classifier(x)
        # out = self.sigmoid(out)

        return out


class UNetTrainer:
    """
    Class for UNet training
    """
    def __init__(self, model_, optimizer, criterion, device='cpu', save_directory=r"", save_step=1, verbose=True):
        """

        Args:
            model_: model to train
            optimizer: model optimizer
            criterion: loss function for this model
            device: training device. Default: cpu
            save_directory: Path for this training session directory. Default: ""
            save_step: Step between epoch saves. Default: 1
            verbose: If 'True', will print verbose output of the model
        """
        self.model = model_
        self.optimizer = optimizer
        self.criterion = criterion
        device = torch.device(device)
        self.device = device
        self.criterion.to(self.device)

        self.train_history_epochs = []
        self.val_history_epochs = []

        self.train_history_batches = []
        self.val_history_batches = []

        self.save_directory = save_directory
        self.save_step = save_step
        self.verbose = verbose

    def train(self, train_dataloader, val_dataloader, epoch_num=5):
        """
        Train function for UNet
        Args:
            train_dataloader: dataloader for training
            val_dataloader: dataloader for validation
            epoch_num: number of epoch for training and validation
        """
        start = datetime.datetime.now()
        print("start time", start.strftime("%d-%m-%Y %H:%M"))

        for epoch in range(epoch_num):
            # Train
            train_loss = self._train_epoch(train_dataloader)
            self.train_history_epochs.append(train_loss)
            if self.verbose:
                print(f'Epoch {epoch} of {epoch_num - 1}, train loss: {train_loss:.5f}')
                now = datetime.datetime.now()
                print("Epoch end time", now.strftime("%d-%m-%Y %H:%M"))
            torch.cuda.empty_cache()

            # Validation
            val_loss = self._val_epoch(val_dataloader)
            self.val_history_epochs.append(val_loss)
            if self.verbose:
                print(f'Epoch {epoch} of {epoch_num - 1}, val loss: {val_loss:.5f}')
                now = datetime.datetime.now()
                print("Epoch end time", now.strftime("%d-%m-%Y %H:%M"))
            torch.cuda.empty_cache()

            # Save
            if self.save_step == 0 or self.save_directory == "":
                continue
            if (epoch + 1) % self.save_step == 0:
                save_model(model.to('cpu'), path=f"{self.save_directory}\\unet_epoch_{self._epoch_string(epoch, epoch_num)}.pt")

        now = datetime.datetime.now()
        print("end time", now.strftime("%d-%m-%Y %H:%M"))
        print("delta", now - start)

    def _train_epoch(self, train_dataloader):
        """
        Method for epoch training
        Args:
            train_dataloader:  train dataloader

        Returns: train loss

        """
        self.model = self.model.to(self.device)

        train_running_loss = 0.0
        self.model.train()
        for data, targets in tqdm(train_dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(data)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            loss_number = loss.item()
            train_running_loss += loss_number
            self.train_history_batches.append(loss_number)

        train_loss = train_running_loss / len(train_dataloader)
        return train_loss

    def _val_epoch(self, val_dataloader):
        """
        Method for epoch validation
        Args:
            val_dataloader:

        Returns: validation loss

        """
        val_running_loss = 0.0
        model.eval()
        for data, targets in tqdm(val_dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            outputs = model(data)

            loss = self.criterion(outputs, targets)
            loss_number = loss.item()
            val_running_loss += loss_number
            self.val_history_batches.append(loss_number)

        val_loss = val_running_loss / len(val_dataloader)
        return val_loss

    @staticmethod
    def _epoch_string(epoch, epoch_num):
        """
        Method to create a string form of current epoch number, using the same number
        of digits for every training session

        Args:
            epoch: number of current epoch
            epoch_num: number of epochs for this training session

        Returns:

        """
        num_digits_epoch_num = _get_num_digits(epoch_num)
        num_digits_epoch = _get_num_digits(epoch)

        epoch_string = "0"*(num_digits_epoch_num - num_digits_epoch) + str(epoch)
        return epoch_string

    def draw_history_plots(self, ):
        """
        Draws plots of train and validation
        """
        pass


def train_unet(model, train_dataloader, val_dataloader, optimizer, device='cpu', epoch_num=5, save_directory=""):
    """
    DEPRECATED

    Function for training and validation segmentation model
    Args:
        model: segmentation model for training
        train_dataloader: dataloader of train dataset
        val_dataloader: dataloader of val dataset
        optimizer: optimizer of the model
        device: device on which calculations will be performed
        epoch_num: number of training epochs
        save_directory: path out for save model weights
    Returns:

    DEPRECATED
    """
    device = torch.device(device)

    train_history = []
    val_history = []

    criterion = FocalLoss()  # nn.CrossEntropyLoss()  # IoULoss
    criterion.to(device)

    for epoch in range(epoch_num):
        model = model.to(device)

        train_running_loss = 0.0
        model.train()
        for data, targets in train_dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            outputs = torch.transpose(outputs, 1, 3)
            outputs = torch.transpose(outputs, 1, 2)

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

            outputs = model(data)
            outputs = torch.transpose(outputs, 1, 3)
            outputs = torch.transpose(outputs, 1, 2)

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
