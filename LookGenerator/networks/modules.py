import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, activation_func=True):
        super(Conv3x3, self).__init__()
        self.net = self._init_net(in_channels, out_channels,
                                  dropout, batch_norm, bias, activation_func)

    def _init_net(self, in_channels, out_channels, dropout, batch_norm, bias, have_activation_func):
        net_list = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, bias=bias)]
        if dropout:
            net_list.append(nn.Dropout2d(p=0.33))
        if batch_norm:
            net_list.append(nn.BatchNorm2d(out_channels))
        if have_activation_func:
            net_list.append(nn.ReLU())
        net = nn.Sequential(*net_list)
        return net

    def init_weights(self, bias):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if bias:
                    nn.init.constant(module.bias, 0)

    def forward(self, x):
        return self.net(x)


class ConvTranspose3x3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, activation_func=True):
        super(ConvTranspose3x3, self).__init__()
        self.net = self._init_net(in_channels, out_channels,
                                  dropout, batch_norm, bias, activation_func)

    def _init_net(self, in_channels, out_channels, dropout, batch_norm, bias, have_activation_func):
        net_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, bias=bias)]
        if dropout:
            net_list.append(nn.Dropout2d(p=0.33))
        if batch_norm:
            net_list.append(nn.BatchNorm2d(out_channels))
        if have_activation_func:
            net_list.append(nn.ReLU)
        net = nn.Sequential(*net_list)
        return net

    def init_weights(self, bias):
        pass

    def forward(self, x):
        return self.net(x)


class Conv5x5(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(Conv5x5, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            Conv3x3(in_channels, out_channels, dropout, batch_norm, bias),
            Conv3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))


class ConvTranspose5x5(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvTranspose5x5, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvTranspose3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvTranspose3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))


class Conv7x7(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(Conv7x7, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            Conv3x3(in_channels, out_channels, dropout, batch_norm, bias),
            Conv3x3(in_channels, out_channels, dropout, batch_norm, bias),
            Conv3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))


class ConvTranspose7x7(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvTranspose7x7, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvTranspose3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvTranspose3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvTranspose3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))