import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LookGenerator.networks.modules import Conv3x3, Conv5x5


class EncoderDecoderGenerator(nn.Module):
    """Generator part of the GAN """

    def __init__(self, clothes_feature_extractor, in_channels=3, out_channels=3, final_activation_func=nn.Sigmoid()):
        """

        Args:
            clothes_feature_extractor: clothes feature extractor for this generation model,
            must be pretrained
            in_channels: input image channels num
            out_channels: output image channels num
        """
        super(EncoderDecoderGenerator, self).__init__()

        self.clothes_feature_extractor = clothes_feature_extractor

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_module1 = nn.Sequential(
            Conv5x5(in_channels=in_channels, out_channels=64, activation_func=nn.LeakyReLU(), res_conn=True)
        )

        self.conv_module2 = Conv3x3(in_channels=64, out_channels=128, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module3 = Conv3x3(in_channels=128, out_channels=256, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module4 = Conv3x3(in_channels=256, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())
        self.conv_module5 = Conv3x3(in_channels=512, out_channels=512, batch_norm=True, activation_func=nn.LeakyReLU())

        self.bottle_neck = Conv3x3(in_channels=512 + self.clothes_feature_extractor.latent_dim_size, out_channels=512,
                                   batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module1 = Conv3x3(in_channels=512 * 2, out_channels=512,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module2 = Conv3x3(in_channels=512 * 2, out_channels=256,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module3 = Conv3x3(in_channels=256 * 2, out_channels=128,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv_conv_module4 = Conv3x3(in_channels=128 * 2, out_channels=64,
                                           batch_norm=True, activation_func=nn.LeakyReLU())

        self.deconv_module5 = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv_conv_module5 = nn.Sequential(
            Conv5x5(in_channels=64 * 2, out_channels=32, batch_norm=True, activation_func=nn.LeakyReLU(), res_conn=False)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1)
        )
        self.final_activation_func = final_activation_func

    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Tensor of packed decoded human and clothes mask.
            First 3 channels for decoded human
        """
        self.clothes_feature_extractor.eval()
        clothes_tensor = x[:, 3:6, :, :]
        out = x[:, 0:3, :, :]

        skip_connections = []

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
        out = self.final_activation_func(out)

        return out

    def __str__(self):
        features_str = str(self.features)
        latent_size_str = str(self.latent_dim_size)

        description = f"EncoderDecoderGenerator:\n" \
                      f"\tfeatures: {features_str}\n" \
                      f"\tlatent_size: {latent_size_str}\n" \
                      f"\tfinal_activation_func: {str(self.final_activation_func)}"
        return description


# TODO: new generator network
class ResNetGenerator(nn.Module):
    def __init__(self):
        pass


class Discriminator(nn.Module):
    """
    Discriminator of the GAN network
    """
    def __init__(self, in_channels=3, batch_norm=False, dropout=True, sigmoid=True):
        """

        Args:
            in_channels: number of input channels
            sigmoid: if 'True', puts the sigmoid activation function in the end
        """
        super(Discriminator, self).__init__()

        self.batch_norm = batch_norm

        self.features = nn.Sequential(
            # input size: in_channels x 256x192
            Conv5x5(
                in_channels=in_channels, out_channels=16,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=16, out_channels=16,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 16x128x96

            Conv5x5(
                in_channels=16, out_channels=32,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=32, out_channels=32,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 32x64x48

            Conv5x5(
                in_channels=32, out_channels=64,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=64, out_channels=64,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 64x32x24

            Conv5x5(
                in_channels=64, out_channels=128,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=128, out_channels=128,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # size: 128x16x12

            Conv5x5(
                in_channels=128, out_channels=256,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=256, out_channels=256,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size: 256x8x6

            Conv5x5(
                in_channels=256, out_channels=512,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=512, out_channels=512,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size: 512x4x3

            Conv5x5(
                in_channels=512, out_channels=1024,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            Conv5x5(
                in_channels=1024, out_channels=1024,
                batch_norm=batch_norm, dropout=dropout,
                activation_func=nn.LeakyReLU(), res_conn=True
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size: 1024x2x1

        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024*2*1, out_features=1)
        )

        self.sigmoid = sigmoid

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        if self.sigmoid:
            out = nn.functional.sigmoid(out)
        return out

    def __str__(self):
        description = f"Discriminator:" \
                      f"\tfeature_extractor: {str(self.features)}\n" \
                      f"\tclassifier: {str(self.classifier)}\n" \
                      f"\tbatch_norm: {self.batch_norm}\n" \
                      f"\tsigmoid: {self.sigmoid}"
        return description

