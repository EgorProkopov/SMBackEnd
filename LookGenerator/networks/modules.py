import torch
import torch.nn as nn
import torch.optim as optim


class ConvModule3x3(nn.Module):
    def __init__(self, in_channels, have_bias=True):
        super(ConvModule3x3, self).__init__()
        self.init_net(in_channels, have_bias)
    #   self.init_weights(have_bias)  # - пока не трогать

    def init_net(self, in_channels, have_bias):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU()
        )

    def init_weights(self, have_bias):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if have_bias:
                    nn.init.constant(module.bias, 0)


    def forward(self, x):
        return self.net(x)


class ConvModule5x5(ConvModule3x3):
    def __init__(self, in_channels, have_dropout=False, have_bias=True):
        super(ConvModule5x5, self).__init__(in_channels, have_bias)
        self.init_net(in_channels, have_dropout, have_bias)
        #self.init_weights(have_bias)

    def init_net(self, in_channels, have_dropout, have_bias):
        if have_dropout:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.Dropout2d(p=0.33),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.Dropout2d(p=0.33),
                nn.ReLU()
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.ReLU()
            )


class ConvModule5x5BN(ConvModule3x3):
    def __init__(self, in_channels, out_channels=None, have_bias=True):
        super(ConvModule5x5BN, self).__init__(in_channels, have_bias)
        if out_channels == None:
            out_channels = 2*in_channels

        self.init_net(in_channels, out_channels, have_bias)

    def init_net(self, in_channels, out_channels, have_bias):
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=have_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=have_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ConvTransposeModule5x5(nn.Module):
    def __init__(self, in_channels, have_bias=True):
        super(ConvTransposeModule5x5, self).__init__()
        self.init_net(in_channels, have_bias)
        #self.init_weights(have_bias)

    def init_net(self, in_channels, have_bias):
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels/2, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels/2, in_channels/2, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

    def init_weights(self, have_bias):
        pass #TODO: сделать инициализацию весов


class ConvModule7x7(ConvModule3x3):
    def __init__(self, in_channels, have_dropout=False, have_bias=True):
        super(ConvModule7x7, self).__init__(in_channels, have_dropout, have_bias)
        self.init_net(in_channels, have_dropout, have_bias)
        #self.init_weights(have_bias)

    def init_net(self, in_channels, have_dropout, have_bias):
        if have_dropout:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.Dropout2d(p=0.33),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.Dropout2d(p=0.33),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.Dropout2d(p=0.33),
                nn.ReLU()
            )

        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.ReLU(),

                nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1, bias=have_bias),
                nn.ReLU()
            )


class ConvTransposeModule7x7(ConvTransposeModule5x5):
    def __init__(self, in_channels, have_bias=True):
        super(ConvTransposeModule7x7, self).__init__(in_channels, have_bias)
        self.init_net(in_channels, have_bias)
        #self.init_weights(have_bias)

    def init_net(self, in_channels, have_bias):
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels/2, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels/2, in_channels/2, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels/2, in_channels/2, kernel_size=3, padding=1, bias=have_bias),
            nn.ReLU()
        )