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


class PoolingLayer2d(nn.Module):
    def __init__(self, num_channels):
        super(PoolingLayer2d, self).__init__()
        self.pooling_layer = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        return self.pooling_layer(x)


class GatedConv(nn.Module):
    def __init__(
            self, input_channels, out_channels,
            dropout=False, batch_norm=False, instance_norm=False, activation_func=None, bias=True
    ):
        super(GatedConv).__init__()

        self.in_channels = input_channels
        self.out_channels = out_channels

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm

        self.activation_func = activation_func

        self.bias = bias

        self.conv_net, self.conv_mask_net = self.init_net()

    def init_net(self):
        conv_list = [nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=self.bias)]
        if self.dropout:
            conv_list.append(nn.Dropout2d(p=0.33))
        if self.batch_norm:
            conv_list.append(nn.BatchNorm2d(self.out_channels))
        if self.instance_norm:
            conv_list.append(nn.InstanceNorm2d(self.out_channels))
        if self.activation_func:
            conv_list.append(self.activation_func)
        conv_net = nn.Sequential(*conv_list)

        conv_mask_list = [nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=self.bias)]
        if self.dropout:
            conv_list.append(nn.Dropout2d(p=0.33))
        if self.batch_norm:
            conv_list.append(nn.BatchNorm2d(self.out_channels))
        if self.instance_norm:
            conv_list.append(nn.InstanceNorm2d(self.out_channels))

        conv_mask_list.append(nn.Sigmoid())

        conv_mask_net = nn.Sequential(*conv_mask_list)

        return conv_net, conv_mask_net

    def forward(self, x, mask):
        conv_out = self.conv_net(x)
        mask_out = self.conv_mask_net(mask)
        gated_out = conv_out * mask_out
        output = gated_out + x

        return output


class SelfAttentionBlock(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, num_channels, activation_func=nn.LeakyReLU()):
        super(SelfAttentionBlock, self).__init__()
        self.chanel_in = num_channels
        self.activation_func = activation_func

        self.f = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=num_channels, out_channels=num_channels // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        f = self.activation_func(self.f(x)).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.activation_func(self.g(x)).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.activation_func(self.h(x)).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attention = torch.bmm(h, attention)  # B * C * (W * H)
        self_attention = self_attention.view(m_batchsize, C, width, height)  # B * C * W * H

        out = self.gamma * self_attention + x
        return out
