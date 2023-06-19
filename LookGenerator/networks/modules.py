import torch
import torch.nn as nn


class Conv3x3(nn.Module):
    """Convolution module with optional batch_norm, dropout layers and activation function."""
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, instance_norm=False, activation_func=None, bias=True):
        """

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels of the output image
            dropout: if 'True', Then adds dropout layer in this module
            batch_norm: if 'True', then adds batch_norm layer in this module
            bias: if 'True', then  adds a bias to the convolutional layer
            activation_func: Adds activation func
        """
        super(Conv3x3, self).__init__()
        self.net = self._init_net(in_channels, out_channels,
                                  dropout, batch_norm, instance_norm, bias, activation_func)

    def _init_net(self, in_channels, out_channels, dropout, batch_norm, instance_norm, bias, activation_func):
        """
        Initialize module network
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels in the output image
            dropout: if 'True', Then adds dropout layer in this module
            batch_norm: if 'True', then adds batch_norm layer in this module
            bias: if 'True', then  adds a bias to the convolutional layer
            activation_func: Adds activation function

        Returns:
            Module network
        """
        net_list = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)]
        if dropout:
            net_list.append(nn.Dropout2d(p=0.33))
        if batch_norm:
            net_list.append(nn.BatchNorm2d(out_channels))
        if instance_norm:
            net_list.append(nn.InstanceNorm2d(out_channels))
        if activation_func:
            net_list.append(activation_func)
        net = nn.Sequential(*net_list)
        return net

    def init_weights(self, bias):
        """
        Initialize weights of the network
        Args:
            bias:
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if bias:
                    nn.init.constant(module.bias, 0)

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        return self.net(x)


class Conv5x5(nn.Module):
    """
    Convolution module that replaces conv layer with kernel_size=5 by 2 conv layers with kernel_size=3.
    Have optional Dropout and BatchNorm layers after every conv layer and optional residual connection.
    """
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, instance_norm=False, activation_func=None, bias=True, res_conn=False):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels in the output image
            dropout: if 'True', Then adds dropout layer in this module
            batch_norm: if 'True', then adds batch_norm layer in this module
            bias: if 'True', then  adds a bias to the convolutional layers
            res_conn: if 'True', then adds residual connection through layers in this module
        """
        super(Conv5x5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_conn = res_conn
        self.net = nn.Sequential(
            Conv3x3(
                self.in_channels, self.out_channels,
                dropout=dropout, batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func, bias=bias
            ),
            Conv3x3(
                self.out_channels, self.out_channels,
                dropout=dropout, batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func, bias=bias
            )
        )
        self._activation_func = activation_func

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        out = self.net(x)
        if self.res_conn:
            shortcut = x
            additional_channels = torch.zeros((
                shortcut.shape[0], abs(self.out_channels - self.in_channels), shortcut.shape[2], shortcut.shape[3]
            ))
            additional_channels = additional_channels.to(shortcut.device)
            shortcut = torch.cat((shortcut, additional_channels), dim=1)
            out = out + shortcut

        if self._activation_func:
            return self._activation_func(out)
        else:
            return out


class Conv7x7(nn.Module):
    """
    Convolution module that replaces conv layer with kernel_size=7 by 3 conv layers with kernel_size=3.
    Have optional Dropout and BatchNorm layers after every conv layer and optional residual connection.
    """
    def __init__(self, in_channels, out_channels,
                 dropout=False, batch_norm=False, instance_norm=False, activation_func=None, bias=True, res_conn=False):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels in the output image
            dropout: if 'True', Then adds dropout layer in this module
            batch_norm: if 'True', then adds batch_norm layer in this module
            bias: if 'True', then  adds a bias to the convolutional layers
            res_conn: if 'True', then adds residual connection through layers in this module
        """
        # res_conn == skip_conn ?
        super(Conv7x7, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_conn = res_conn
        self.net = nn.Sequential(
            Conv3x3(
                self.in_channels, self.out_channels,
                dropout=dropout, batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func, bias=bias
            ),
            Conv3x3(
                self.out_channels, self.out_channels,
                dropout=dropout, batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func, bias=bias
            ),
            Conv3x3(
                self.out_channels, self.out_channels,
                dropout=dropout, batch_norm=batch_norm, instance_norm=instance_norm,
                activation_func=activation_func, bias=bias
            )
        )
        self._activation_func = activation_func

    def forward(self, x):
        """
        Forward propagation method of neural network
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        out = self.net(x)
        if self.res_conn:
            shortcut = x
            additional_channels = torch.zeros((
                shortcut.shape[0], self.out_channels - self.in_channels, shortcut.shape[2], shortcut.shape[3]
            ))
            shortcut = torch.cat((shortcut, additional_channels), dim=1)
            out = out + shortcut

        if self._activation_func:
            return self._activation_func(out)
        else:
            return out
