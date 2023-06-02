import os

import numpy as np
import matplotlib.pyplot as plt
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

    def draw_history_plots(self, epochs=True):
        """
        Draws plots of train and validation

        Args:
            epochs: if 'True', draws history plots by epochs, else by batches
        """
        if epochs:
            plt.plot(self.train_history_epochs, label="train")
            plt.plot(self.val_history_epochs, label="val")
            plt.show()
        else:
            plt.plot(self.train_history_batches, label="train")
            plt.plot(self.val_history_batches, label="val")
            plt.show()

    def save_history_plots(self, save_dir, epochs=True):
        """
        Method to save plots images
        Args:
            save_dir: directory to save plots images
            epochs:  if 'True', saves history plots by epochs, else by batches

        """
        if epochs:
            plt.plot(self.train_history_epochs, label="train")
            plt.plot(self.val_history_epochs, label="val")
            plt.savefig(os.path.join(save_dir, "plot.png"))
        else:
            plt.plot(self.train_history_batches, label="train")
            plt.plot(self.val_history_batches, label="val")
            plt.savefig(os.path.join(save_dir, "plot.png"))

    def create_readme(self, save_dir):
        """
        Method to create readme.txt file with info about trained network
        Args:
            save_dir: directory to save readme.txt file
        """
        readme = repr(self)
        file = open(os.path.join(save_dir, "readme.txt"), 'w')
        file.write(readme)
        file.close()

    def __repr__(self):
        description = f"Model:\n\t{repr(self.model)}\n" \
                      f"Criterion: \n\t{repr(self.criterion)}\n" \
                      f"Optimizer: \n\t{repr(type(self.optimizer))}"
        return description


class TrainerWithMask(Trainer):
    def __init__(
            self, model_,
            optimizer, criterion,
            device='cpu', save_directory=r"",
            save_step=1, verbose=True
    ):
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

        super().__init__(model_, optimizer, criterion, device, save_directory, save_step, verbose)

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
        for data, mask, targets in tqdm(train_dataloader):
            data = data.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(data)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, mask, targets)
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
        for data, mask, targets in tqdm(val_dataloader):
            data = data.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(data)

            loss = self.criterion(outputs, mask, targets)
            loss_number = loss.item()
            val_running_loss += loss_number
            self.val_history_batches.append(loss_number)

        val_loss = val_running_loss / len(val_dataloader)
        return val_loss


class GANTrainer:
    def __init__(
            self, models, optimizers, criterions,
            save_step=1, save_directory_discriminator=r"", save_directory_generator=r"",
            device='cpu', verbose=True
    ):
        device = torch.device(device)
        self.device = device

        self.models = models
        self.optimizers = optimizers
        self.criterions = criterions

        # TODO: сделать сохранение лоссов
        self.discriminator_fake_history_epochs = []
        self.discriminator_real_history_epochs = []
        self.discriminator_history_epochs = []
        self.generator_history_epochs = []

        self.discriminator_fake_history_batches = []
        self.discriminator_real_history_batches = []
        self.discriminator_history_batches = []
        self.generator_history_batches = []

        self.save_directory_discriminator = save_directory_discriminator
        self.save_directory_generator = save_directory_generator
        self.save_step = save_step
        self.verbose = verbose

    def train(self, train_dl, epochs_num=10):
        self.models["discriminator"].train()
        self.models["discriminator"] = self.models["discriminator"].to(self.device)
        self.models["generator"].train()
        self.models["generator"] = self.models["generator"].to(self.device)

        self.criterions["discriminator"] = self.criterions["discriminator"].to(self.device)
        self.criterions["generator"] = self.criterions["generator"].to(self.device)
        torch.cuda.empty_cache()

        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        for epoch in range(epochs_num):
            loss_d_per_epoch = []
            loss_g_per_epoch = []
            real_score_per_epoch = []
            fake_score_per_epoch = []
            for input_images, real_images in tqdm(train_dl):
                input_images = input_images.to(self.device)
                real_images = real_images.to(self.device)
                # Train discriminator
                # Clear discriminator gradients
                self.optimizers["discriminator"].zero_grad()

                real_images = real_images.to(self.device)

                # Pass real images through discriminator
                real_preds = self.models["discriminator"](real_images)
                real_targets = torch.ones(real_images.shape[0], 1, device=self.device)
                real_loss = self.criterions["discriminator"](real_preds, real_targets)
                cur_real_score = torch.mean(real_preds).item()

                # Generate fake images
                fake_images = self.models["generator"](input_images)

                # Pass fake images through discriminator
                fake_targets = torch.ones(fake_images.shape[0], 1, device=self.device)
                fake_preds = self.models["discriminator"](fake_images)
                fake_loss = self.criterions["discriminator"](fake_preds, fake_targets)
                cur_fake_score = torch.mean(fake_preds).item()

                real_score_per_epoch.append(cur_real_score)
                fake_score_per_epoch.append(cur_fake_score)

                # Update discriminator weights
                loss_d = real_loss + fake_loss
                loss_d.backward()
                self.optimizers["discriminator"].step()
                loss_d_per_epoch.append(loss_d.item())

                # Train generator

                # Clear generator gradients
                self.optimizers["generator"].zero_grad()

                # Generate fake images
                fake_images = self.models["generator"](input_images)

                # Try to fool the discriminator
                preds = self.models["discriminator"](fake_images)
                targets = torch.ones(real_images.shape[0], 1, device=self.device)
                loss_g = self.criterions["generator"](preds, targets, fake_images, real_images)

                # Update generator weights
                loss_g.backward()
                self.optimizers["generator"].step()
                loss_g_per_epoch.append(loss_g.item())

                losses_g.append(np.mean(loss_g_per_epoch))

            # Record losses & scores
            losses_d.append(np.mean(loss_d_per_epoch))
            real_scores.append(np.mean(real_score_per_epoch))
            fake_scores.append(np.mean(fake_score_per_epoch))

            # Log losses & scores (last batch)
            if self.verbose:
                print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                    epoch + 1, epochs_num,
                    losses_g[-1], losses_d[-1], real_scores[-1], fake_scores[-1])
                )

            if self.save_step == 0 or self.save_directory_discriminator == "" or self.save_directory_generator == "":
                continue

            if (epoch + 1) % self.save_step == 0:
                save_model(
                    self.models["discriminator"].to('cpu'),
                    path=f"{self.save_directory_discriminator}\\discriminator_epoch_{self._epoch_string(epoch, epochs_num)}.pt"
                )
                save_model(
                    self.models["generator"].to('cpu'),
                    path=f"{self.save_directory_generator}\\generator_epoch_{self._epoch_string(epoch, epochs_num)}.pt"
                )

        return losses_g, losses_d, real_scores, fake_scores

    def draw_plots(self):
        # TODO: перед реализацией функции отрисовки графиков нужно реализовать историю
        pass

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


class WGANGPTrainer:
    """
    TODO: не работает, нужно переписать функцию тренировки
    """
    def __init__(
            self, generator, discriminator,
            optimizer_generator,  optimizer_discriminator,
            criterion_generator, criterion_discriminator, gradient_penalty, gp_weight=0.2,
            save_step=1, save_directory_discriminator=r"", save_directory_generator=r"",
            device='cpu', verbose=True
    ):
        device = torch.device(device)
        self.device = device

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator

        self.criterion_generator = criterion_generator
        self.criterion_discriminator = criterion_discriminator

        self.criterion_generator.to(self.device)
        self.criterion_discriminator.to(self.device)

        self.gradient_penalty = gradient_penalty
        self.gp_weight = gp_weight

        self.discriminator_fake_history_epochs = []
        self.discriminator_real_history_epochs = []
        self.discriminator_history_epochs = []
        self.generator_history_epochs = []

        self.discriminator_fake_history_batches = []
        self.discriminator_real_history_batches = []
        self.discriminator_history_batches = []
        self.generator_history_batches = []

        self.save_directory_discriminator = save_directory_discriminator
        self.save_directory_generator = save_directory_generator
        self.save_step = save_step
        self.verbose = verbose

    def train(self, train_dataloader, epoch_num=5):
        """
        Train function
        Args:
            train_dataloader: dataloader for training
            epoch_num: number of epoch for training and validation
        """
        start = datetime.datetime.now()
        print("start time", start.strftime("%d-%m-%Y %H:%M"))

        for epoch in range(epoch_num):
            # Train epoch

            loss_real, loss_fake, loss_d, loss_g = self._train_epoch(train_dataloader)

            if self.verbose:
                print(f'Epoch {epoch} of {epoch_num - 1}, discriminator loss: {loss_d:.5f}')
                print(f'Epoch {epoch} of {epoch_num - 1}, generator loss: {loss_g:.5f}')
                now = datetime.datetime.now()
                print("Epoch end time", now.strftime("%d-%m-%Y %H:%M"))

            torch.cuda.empty_cache()

            # Save
            # TODO: make a new method to save models
            if self.save_step == 0 or self.save_directory_discriminator == "" or self.save_directory_generator == "":
                continue

            if (epoch + 1) % self.save_step == 0:
                save_model(
                    self.discriminator.to('cpu'),
                    path=f"{self.save_directory_discriminator}\\discriminator_epoch_{self._epoch_string(epoch, epoch_num)}.pt"
                )
                save_model(
                    self.generator.to('cpu'),
                    path=f"{self.save_directory_generator}\\generator_epoch_{self._epoch_string(epoch, epoch_num)}.pt"
                )

        now = datetime.datetime.now()
        print("end time", now.strftime("%d-%m-%Y %H:%M"))
        print("delta", now - start)

    def _train_epoch(self, train_dataloader):
        self.discriminator_real_epoch_batches_loss = []
        self.discriminator_fake_epoch_batches_loss = []
        self.discriminator_epoch_batches_loss = []
        self.generator_epoch_batches_loss = []

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        for iteration, (input_images, real_images) in enumerate(tqdm(train_dataloader), 0):
            input_images = input_images.to(self.device)
            real_images = real_images.to(self.device)
            self._train_discriminator(input_images, real_images)

            if iteration % 5 == 0:
                self._train_generator(input_images, real_images)

        loss_real = np.mean(self.discriminator_real_epoch_batches_loss)
        loss_fake = np.mean(self.discriminator_fake_epoch_batches_loss)
        loss_d = np.mean(self.discriminator_fake_epoch_batches_loss)
        loss_g = np.mean(self.generator_epoch_batches_loss)

        self.discriminator_real_history_epochs.append(loss_real)
        self.discriminator_fake_history_epochs.append(loss_fake)
        self.discriminator_history_epochs.append(loss_d)
        self.generator_history_epochs.append(loss_g)

        return loss_real, loss_fake, loss_d, loss_g

    def _train_discriminator(self, input_images, real_images):
        self.discriminator.train()
        self.generator.eval()

        # Clear discriminator gradients
        self.optimizer_discriminator.zero_grad()

        # Move batch to device
        real_images = real_images.to(self.device)

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.shape[0], 1, device=self.device)
        real_loss = self.criterion_discriminator(real_preds, real_targets)
        self.discriminator_real_history_batches.append(torch.mean(real_loss).item())
        self.discriminator_real_epoch_batches_loss.append(torch.mean(real_loss).item())

        # Generate fake images
        fake_images = self.generator(input_images)

        # Pass fake images through discriminator
        fake_targets = -torch.ones(fake_images.shape[0], 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = self.criterion_discriminator(fake_preds, fake_targets)
        self.discriminator_fake_history_batches.append(torch.mean(fake_loss).item())
        self.discriminator_fake_epoch_batches_loss.append(torch.mean(fake_loss).item())

        # Loss computation
        gp = self.gradient_penalty(self.discriminator, fake_images, real_images, self.device)
        loss_discriminator = real_loss + fake_loss + self.gp_weight * gp
        self.discriminator_history_batches.append(torch.mean(loss_discriminator).item())
        self.discriminator_epoch_batches_loss.append(torch.mean(loss_discriminator).item())

        # Update discriminator weights
        loss_discriminator.backward()
        self.optimizer_discriminator.step()

    def _train_generator(self, input_images, real_images):
        self.discriminator.eval()
        self.generator.train()

        # Clear generator gradients
        self.optimizer_generator.zero_grad()

        # Generate fake images
        fake_images = self.generator(input_images)

        # Try to fool discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(real_images.shape[0], 1, device=self.device)
        loss_g = self.criterion_generator(preds, targets, fake_images, real_images)
        self.generator_history_batches.append(torch.mean(loss_g).item())
        self.generator_epoch_batches_loss.append(torch.mean(loss_g).item())

        # Update generator weights
        loss_g.backward()
        self.optimizer_generator.step()

        # TODO: test it

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

    def draw_history_plots(self):
        """
        Draws history plots
        """
        plt.plot(self.discriminator_real_history_epochs, label="discriminator_real")
        plt.plot(self.discriminator_fake_history_epochs, label="discriminator_fake")
        plt.plot(self.discriminator_history_epochs, label="discriminator_history")
        plt.plot(self.generator_history_epochs, label="generator")
        plt.legend()
        plt.show()

    def save_history_plots(self, save_dir):
        """
        Method to save plots images
        Args:
            save_dir: directory to save plots images
            epochs:  if 'True', saves history plots by epochs, else by batches

        """
        plt.plot(self.discriminator_real_history_epochs, label="discriminator_real")
        plt.plot(self.discriminator_fake_history_epochs, label="discriminator_fake")
        plt.plot(self.discriminator_history_epochs, label="discriminator_history")
        plt.plot(self.generator_history_epochs, label="generator")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "plot.png"))

    def create_readme(self, save_dir):
        """
        Method to create readme.txt file with info about trained network
        Args:
            save_dir: directory to save readme.txt file
        """
        readme = repr(self)
        file = open(os.path.join(save_dir, "readme.txt"), 'w')
        file.write(readme)
        file.close()

    def __repr__(self):
        description = f"Generator:\n\t{repr(self.generator)}\n" \
                      f"Discriminator:\n\t{repr(self.discriminator)}\n" \
                      f"Criterion generator: \n\t{repr(self.criterion_generator)}\n" \
                      f"Criterion discriminator: \n\t{repr(type(self.criterion_discriminator))}\n" \
                      f"Optimizer generator: \n\t{repr(self.optimizer_generator)}\n" \
                      f"Optimizer discriminator: \n\t{repr(type(self.optimizer_discriminator))}\n"
        return description


