import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv3x3


# TODO: test it
class EncoderDecoder(nn.Module):
    """Model of encoder-decoder part of virtual try-on model"""
    def __init__(self, in_channels=22, out_channels=4):
        super(EncoderDecoder, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = Conv3x3(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU())
        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        self.bottle_neck = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512*2, out_channels=512, batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512*2, out_channels=256, batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256*2, out_channels=128, batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128*2, out_channels=64, batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.deconv_conv_module5 = Conv3x3(in_channels=64*2, out_channels=32, batch_norm=True, activation_func=nn.ReLU())

        self.final_conv = Conv3x3(in_channels=32, out_channels=out_channels, batch_norm=True, activation_func=nn.Tanh())

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Decoded human and mask of clothes
        """
        skip_connections = []
        
        return human_out, clothes_mask_out


def train_encoder_decoder(model, train_dataloader, val_dataloader, optimizer, device='cpu', epoch_num=5):
    pass