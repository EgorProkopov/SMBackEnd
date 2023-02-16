import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv5x5


# TODO: test it
class EncoderDecoder(nn.Module):
    """Model of encoder-decoder part of virtual try-on model"""
    def __init__(self, in_channels=22, out_channels=4):
        super(EncoderDecoder, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.conv1 = Conv5x5(in_channels=in_channels, out_channels=64, )


    def forward(self, x):
        """
        Forward propagation method of neural network.
        Args:
            x: mini-batch of data

        Returns:
            Result of network working
        """
        pose_pack = x[0]
        clothes = x[1]

        pose_out = self.pose_encoder(pose_pack)
        clothes_out = self.clothes_encoder(clothes)

        out = torch.cat((pose_out, clothes_out), dim=1)

        skip_connections = [out]

        for encoder_module in self.encoder_list:
            out = encoder_module(out)
            skip_connections.append(out)

        out = self.bottle_neck(out)
        skip_connections = skip_connections[::-1]

        for decoder_module, skip_connection in zip(self.decoder_list, skip_connections):
            out = torch.cat((out, skip_connection), dim=1)
            out = decoder_module(out)

        human_out = out[0:60]
        clothes_mask_out = out[61:64]

        human_out = self.human_decoder(human_out)
        clothes_mask_out = self.clothes_mask_decoder(clothes_mask_out)

        return human_out, clothes_mask_out


def train_encoder_decoder(model, train_dataloader, val_dataloader, optimizer, device='cpu', epoch_num=5):
    pass