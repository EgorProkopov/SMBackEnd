import torch
import torch.nn as nn


class ConvEncode3x3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, activation_func=True):
        super(ConvEncode3x3, self).__init__()
        self.net = self._init_net(in_channels, out_channels,
                                  dropout, batch_norm, bias, activation_func)

    def _init_net(self, in_channels, out_channels, dropout, batch_norm, bias, have_activation_func):
        net_list = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)]
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


class ConvDecode3x3(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, activation_func=True):
        super(ConvDecode3x3, self).__init__()
        self.net = self._init_net(in_channels, out_channels,
                                  dropout, batch_norm, bias, activation_func)

    def _init_net(self, in_channels, out_channels, dropout, batch_norm, bias, have_activation_func):
        net_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)]
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


class ConvEncode5x5(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvEncode5x5, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvEncode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvEncode3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )
        self._ReLU = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.skip_conn:
            shortcut = x
            return self._ReLU(out + shortcut)
        return self._ReLU(out)


class ConvDecode5x5(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvDecode5x5, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvDecode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvDecode3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))


class ConvEncode7x7(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvEncode7x7, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvEncode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvEncode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvEncode3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))


class ConvDecode7x7(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, bias=True, skip_conn=False):
        super(ConvDecode7x7, self).__init__()
        self.skip_conn = skip_conn
        self.net = nn.Sequential(
            ConvDecode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvDecode3x3(in_channels, out_channels, dropout, batch_norm, bias),
            ConvDecode3x3(out_channels, out_channels, dropout, batch_norm, bias, activation_func=False)
        )

    def forward(self, x):
        if self.skip_conn:
            shortcut = x
            out = self.net(x)
            return nn.ReLU(out + shortcut)
        else:
            return nn.ReLU(self.net(x))