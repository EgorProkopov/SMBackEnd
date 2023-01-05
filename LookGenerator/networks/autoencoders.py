import torch
import torch.nn as nn
import torch.optim as optim

from modules import ConvModule7x7, ConvTransposeModule7x7


class ClothesConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ClothesConvAutoEncoder).__init__()
        self.inin_net()
        #self.init_weights()

    def init_net(self):
        self.init_encoder()
        self.init_decoder()
        self.net = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.33),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.33),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=32, have_dropout=True),
            ConvModule7x7(in_channels=64, have_dropout=True),
            ConvModule7x7(in_channels=128, have_dropout=True),
            ConvModule7x7(in_channels=256, have_dropout=True)
        )

    def init_decoder(self):
        self.decoder = nn.Sequential(
            ConvTransposeModule7x7(in_channels=512, have_dropout=True),
            ConvTransposeModule7x7(in_channels=256, have_dropout=True),
            ConvTransposeModule7x7(in_channels=128, have_dropout=True),
            ConvTransposeModule7x7(in_channels=64, have_dropout=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, in_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def init_weights(self):
        pass

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return self.encoder(x)

    def train_and_val(self, dataloader, epoch_num=10):
        pass


class SourceConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ClothesConvAutoEncoder).__init__()
        self.inin_net()
        #self.init_weights()

    def init_net(self):
        self.init_encoder()
        self.init_decoder()

    # Возможна проблема затухания градиентов
    def init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.33),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.33),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=32, have_dropout=True),
            ConvModule7x7(in_channels=64, have_dropout=True),
            ConvModule7x7(in_channels=128, have_dropout=True),
            ConvModule7x7(in_channels=256, have_dropout=True),
            ConvModule7x7(in_channels=512, have_dropout=True), #out: 1024 channels

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.33),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.33),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

    # Возможна проблема переобучения декодера, нужна регуляризация
    def init_decoder(self):
        self.decoder = nn.Sequential(
            ConvTransposeModule7x7(in_channels=1024),
            ConvTransposeModule7x7(in_channels=512),
            ConvTransposeModule7x7(in_channels=256),
            ConvTransposeModule7x7(in_channels=128),
            ConvTransposeModule7x7(in_channels=64),

            nn.MaxUnpool2d(kernel_size=2, stride=2),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, in_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, in_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, in_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def init_weights(self):
        pass

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return self.encoder(x)

    def train_and_val(self, dataloader, epoch_num=10):
        pass

