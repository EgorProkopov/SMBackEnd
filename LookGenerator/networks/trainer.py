import datetime
from tqdm import tqdm

import torch

from LookGenerator.networks.utils import get_num_digits, save_model


class Trainer:
    """
    Class for model training
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
        Train function
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
                save_model(self.model.to('cpu'), path=f"{self.save_directory}\\epoch_{self._epoch_string(epoch, epoch_num)}.pt")

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
        self.model.eval()
        for data, targets in tqdm(val_dataloader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(data)

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

        Returns: converted to string epoch number

        """
        num_digits_epoch_num = get_num_digits(epoch_num)
        num_digits_epoch = get_num_digits(epoch)

        epoch_string = "0"*(num_digits_epoch_num - num_digits_epoch) + str(epoch)
        return epoch_string

    def draw_history_plots(self, ):
        """
        Draws plots of train and validation
        """
        pass
