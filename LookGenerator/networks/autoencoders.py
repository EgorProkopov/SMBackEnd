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
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=64, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=128, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=256, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def init_decoder(self):
        self.decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=512, have_dropout=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=256, have_dropout=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=128, have_dropout=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=64, have_dropout=True),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def init_weights(self):
        pass

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return self.encoder(x)

    def train_and_val(self, train_dataloader, val_datalaoder, epoch_num=10):
        train_history = []
        val_history = []

        optimizer = optim.Adam(self.net.parameters())
        criterion = nn.MSELoss()

        for epoch in range(epoch_num):
            train_running_loss = 0.0
            for data in train_dataloader:
                outputs = self.net(data)
                optimizer.zero_grad()
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()

            train_loss = train_running_loss/len(train_dataloader)
            train_history.append(loss)
            print(f'Epoch {epoch} of {epoch_num}, train loss: {train_loss:.3f}')

            val_running_loss = 0.0
            for data in val_datalaoder:
                outputs = self.net(data)
                loss = criterion(outputs, data)
                val_running_loss = loss.item()

            val_loss = val_running_loss/len(val_datalaoder)
            val_history.append(val_loss)
            print(f'Epoch {epoch} of {epoch_num}, val loss: {val_loss:.3f}')


class SourceConvAutoEncoder(nn.Module):
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
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=64, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=128, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=256, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvModule7x7(in_channels=512, have_dropout=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #out: 1024 channels

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
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=1024),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=512),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=256),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=128),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTransposeModule7x7(in_channels=64),

            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def init_weights(self):
        pass

    def forward(self, x):
        return self.net(x)

    def encode(self, x):
        return self.encoder(x)

    def train_and_val(self, train_dataloader, val_datalaoder, epoch_num=10):
        train_history = []
        val_history = []

        optimizer = optim.Adam(self.net.parameters())
        criterion = nn.MSELoss()

        for epoch in range(epoch_num):
            train_running_loss = 0.0
            for data in train_dataloader:
                outputs = self.net(data)
                optimizer.zero_grad()
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item()

            train_loss = train_running_loss/len(train_dataloader)
            train_history.append(loss)
            print(f'Epoch {epoch} of {epoch_num}, train loss: {train_loss:.3f}')

            val_running_loss = 0.0
            for data in val_datalaoder:
                outputs = self.net(data)
                loss = criterion(outputs, data)
                val_running_loss = loss.item()

            val_loss = val_running_loss/len(val_datalaoder)
            val_history.append(val_loss)
            print(f'Epoch {epoch} of {epoch_num}, val loss: {val_loss:.3f}')

