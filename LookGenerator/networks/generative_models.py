import torch
import torch.nn as nn

from LookGenerator.networks.modules import Conv3x3, Conv5x5, ConvTranspose5x5, Conv7x7, ConvTranspose7x7


# TODO: test it
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.pose_encoder = nn.Sequential(
            Conv5x5(in_channels=22, out_channels=30, batch_norm=True, skip_conn=True),
            Conv5x5(in_channels=30, out_channels=60, batch_norm=True, skip_conn=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.clothes_encoder = nn.Sequential(
            Conv5x5(in_channels=3, out_channels=4, batch_norm=True, skip_conn=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder_list = [
            nn.Sequential(
                Conv5x5(in_channels=64, out_channels=128, batch_norm=True, skip_conn=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),                                          # in:128x96  out:64x48
            nn.Sequential(
                Conv5x5(in_channels=128, out_channels=256, batch_norm=True, skip_conn=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),                                          # in:64x48  out:32x24
            nn.Sequential(
                Conv5x5(in_channels=256, out_channels=512, batch_norm=True, skip_conn=True),
                nn.MaxPool2d(kernel_size=4, stride=4)
            )                                           # in:32x24  out:8x6
        ]

        self.bottle_neck = nn.Sequential(
            Conv5x5(in_channels=512, out_channels=512, batch_norm=True, skip_conn=True)         # in:8x6  out:8x6
        )

        self.decoder_list = [
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=4, stride=4),                                        # in:8x6  out:32x24
                ConvTranspose5x5(in_channels=512+512, out_channels=256, batch_norm=True, skip_conn=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),                                        # in:32x24  out:64x48
                ConvTranspose5x5(in_channels=256+256, out_channels=128, batch_norm=True, skip_conn=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),                                        # in:64x48  out:128x96
                ConvTranspose5x5(in_channels=128+128, out_channels=64, batch_norm=True, skip_conn=True)
            )
        ]

        self.human_decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTranspose5x5(in_channels=60, out_channels=20, batch_norm=True, skip_conn=True),
            Conv5x5(in_channels=20, out_channels=3, batch_norm=True, skip_conn=True)
        )

        self.clothes_mask_decoder = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvTranspose5x5(in_channels=4, out_channels=2, batch_norm=True, skip_conn=True),
            Conv3x3(in_channels=2, out_channels=1, batch_norm=True)
        )

    def forward(self, x):
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

