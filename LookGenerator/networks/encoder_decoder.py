import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LookGenerator.networks.modules import Conv3x3, Conv5x5
from LookGenerator.networks.utils import save_model


class EncoderDecoder(nn.Module):
    """Model of encoder-decoder part of virtual try-on model"""
    def __init__(self, clothes_feature_extractor, in_channels=22, out_channels=3):
        super(EncoderDecoder, self).__init__()

        self.clothes_feature_extractor = clothes_feature_extractor

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = nn.Sequential(
            Conv5x5(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU()),
            Conv5x5(in_channels=64, out_channels=64, activation_func=nn.LeakyReLU())
        )

        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        self.bottle_neck = Conv3x3(in_channels=512 + self.clothes_feature_extractor.latent_dim_size, out_channels=512,
                                   batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512*2, out_channels=512,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512*2, out_channels=256,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256*2, out_channels=128,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128*2, out_channels=64,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv_conv_module5 = nn.Sequential(
            Conv5x5(in_channels=64*2, out_channels=32, batch_norm=True, activation_func=nn.LeakyReLU()),
            Conv5x5(in_channels=32, out_channels=32, batch_norm=True, activation_func=nn.LeakyReLU())
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Tensor of packed decoded human and clothes mask.
            First 3 channels for decoded human
        """
        skip_connections = []

        self.clothes_feature_extractor.eval()
        clothes_tensor = x[:, 3:6, :, :]
        out = x[:, 0:3, :, :]

        out = self.conv_module1(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module2(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module3(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module4(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        out = self.conv_module5(out)
        skip_connections.append(out)
        out = self.max_pool(out)

        clothes_features = self.clothes_feature_extractor.encode(clothes_tensor)
        clothes_features = transforms.functional.resize(clothes_features, size=out.shape[3:])

        out = torch.concat((out, clothes_features), axis=1)
        out = self.bottle_neck(out)

        out = self.deconv_module1(out)
        out = torch.cat((out, skip_connections[4]), axis=1)
        out = self.deconv_conv_module1(out)

        out = self.deconv_module2(out)
        out = torch.cat((out, skip_connections[3]), axis=1)
        out = self.deconv_conv_module2(out)

        out = self.deconv_module3(out)
        out = torch.cat((out, skip_connections[2]), axis=1)
        out = self.deconv_conv_module3(out)

        out = self.deconv_module4(out)
        out = torch.cat((out, skip_connections[1]), axis=1)
        out = self.deconv_conv_module4(out)

        out = self.deconv_module5(out)
        out = torch.cat((out, skip_connections[0]), axis=1)
        out = self.deconv_conv_module5(out)

        out = self.final_conv(out)

        return out
