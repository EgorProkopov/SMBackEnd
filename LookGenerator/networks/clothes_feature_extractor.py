import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv3x3, Conv5x5
from LookGenerator.networks.utils import save_model


class ClothingAutoEncoder(nn.Module):
    """Model of encoder-decoder part of virtual try-on model"""
    # TODO: написать ввод кол-ва карт активаций в аргументах
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        """

        Args:
            in_channels: number of channels of input image
            out_channels: number of channels of output image
            latent_dim: latent dimension size
        """
        super(ClothingAutoEncoder, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = nn.Sequential(
            Conv5x5(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU(), batch_norm=True),
            Conv5x5(in_channels=64, out_channels=64, activation_func=nn.LeakyReLU(), batch_norm=True)
        )

        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        #512 x 8 x 6
        self.bottle_neck = Conv3x3(in_channels=512, out_channels=512,
                                   batch_norm=True, activation_func=nn.LeakyReLU())
        # 512 x 8 x 6
        self.latent_dim = latent_dim
        self.latent_linear = nn.Linear(512 * 8 * 6, self.latent_dim)

        self.mu = nn.Linear(512 * 8 * 6, self.latent_dim)
        self.log_var = nn.Linear(512 * 8 * 6, self.latent_dim)

        self.decode_input = nn.Linear(self.latent_dim, 512 * 8 * 6)

        self.deconv_module1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512, out_channels=512,
                                           batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512, out_channels=256,
                                           batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256, out_channels=128,
                                           batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128, out_channels=64,
                                           batch_norm=True, activation_func=nn.ReLU())

        self.deconv_module5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv_conv_module5 = nn.Sequential(
            Conv5x5(in_channels=64, out_channels=32, batch_norm=True, activation_func=nn.ReLU()),
            Conv5x5(in_channels=32, out_channels=out_channels, batch_norm=True, activation_func=nn.ReLU())
        )

        # self.final_conv = Conv3x3(in_channels=32, out_channels=out_channels,
        #                           batch_norm=True, activation_func=nn.ReLU())

        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
        #     nn.Sigmoid()
        # )
    def encode(self, x):
        x = self.conv_module1(x)
        x = self.max_pool(x)

        x = self.conv_module2(x)
        x = self.max_pool(x)

        x = self.conv_module3(x)
        x = self.max_pool(x)

        x = self.conv_module4(x)
        x = self.max_pool(x)

        x = self.conv_module5(x)
        x = self.max_pool(x)

        x = self.bottle_neck(x)

        x = self.latent_linear(x)

        # mu = self.mu(x)
        # log_var = self.log_var(x)

        return x

    def _sampler(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.distributions.Normal(0, 1).sample()
        return std * eps + mu

    def extract_features(self, x):
        mu, log_var = self.encode(x)
        sample = self._sampler(mu, log_var)
        features = self.decode_input(sample)
        features = torch.reshape(features, (-1, 512, 8, 6))
        return features

    def _decode(self, z):
        out = self.decode_input(z)
        out = torch.reshape(out, (-1, 512, 8, 6))

        out = self.deconv_module1(out)
        out = self.deconv_conv_module1(out)

        out = self.deconv_module2(out)
        out = self.deconv_conv_module2(out)

        out = self.deconv_module3(out)
        out = self.deconv_conv_module3(out)

        out = self.deconv_module4(out)
        out = self.deconv_conv_module4(out)

        out = self.deconv_module5(out)
        out = self.deconv_conv_module5(out)

        # z = self.final_conv(z)
        return out

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Tensor of packed decoded human and clothes mask.
            First 3 channels for decoded human
        """
        #mu, log_var = self._encode(x)
        #z = self._sampler(mu, log_var)

        to_decoder = self.encode(x)
        out = self._decode(to_decoder)

        return out


class ClothAutoencoder(nn.Module):
    """
    Autoencoder for cloth with changeable layers number and size
    """
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            features=(8, 16, 32, 64),
            latent_dim_size=128,
            encoder_activation_func=nn.LeakyReLU(),
            decoder_activation_func=nn.ReLU()
    ):
        """

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            features: tuple of encoding layers number and size
            latent_dim_size: size of latent dim
            encoder_activation_func: activation function for encoder layers
            decoder_activation_func: activation function for decoder layers
        """
        super(ClothAutoencoder, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(
                Conv3x3(
                    in_channels, feature,
                    batch_norm=True, dropout=False,
                    activation_func=encoder_activation_func
                ))
            in_channels = feature
            self.encoder.append(self.max_pool)

        self.bottle_neck = Conv3x3(
            in_channels=in_channels,
            out_channels=latent_dim_size
        )
        in_channels = latent_dim_size

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                Conv3x3(
                    in_channels=in_channels,
                    out_channels=feature,
                    batch_norm=True, dropout=False,
                    activation_func=decoder_activation_func
                )
            )
            self.decoder.append(self.upsampling)
            in_channels = feature

        self.final_conv = Conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            batch_norm=True,
            dropout=False,
            activation_func=nn.Sigmoid()
        )

    def encode(self, x):
        """

        Encoding of input image

        Args:
            x: input image

        Returns: encoded image

        """
        out = self.encoder(x)
        out = self.bottle_neck(out)
        return out

    def decode(self, z):
        """
        Decoding of encoded image or noise (latent space embedding)
        Args:
            z: embedding from latent space

        Returns: decoded image

        """
        out = self.decoder(z)
        out = self.final_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out
