from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch

from LookGenerator.networks.clothes_feature_extractor import ClothAutoencoder
from LookGenerator.networks.encoder_decoder import EncoderDecoder


def test_output_shape():
    input_ = torch.ones((1, 6, 256, 192))

    out_channels = 3
    clothes_autoencoder = ClothAutoencoder()
    model = EncoderDecoder(
        clothes_feature_extractor=clothes_autoencoder,
        in_channels=3,
        out_channels=out_channels
    )
    output = model(input_)
    plt.imshow(output.detach().numpy()[0, 0, :, :])
    plt.show()
    plt.imshow(output.detach().numpy()[0, 1, :, :])
    plt.show()
    plt.imshow(output.detach().numpy()[0, 2, :, :])
    plt.show()
    print(output.shape)
    assert output.shape == (1, out_channels, 256, 192), "Test 1 Failed"
    print("Test 1 Complete")


if __name__ == "__main__":
    test_output_shape()
