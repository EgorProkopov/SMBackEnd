from PIL import Image
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch

from LookGenerator.networks.seg_gen import TwoHeadUNet


def test_output_shape():
    input_x = torch.ones((1, 15, 256, 192))
    input_clothes = torch.ones((1, 3, 256, 192))
    out_channels = 1

    model = TwoHeadUNet(
        in_channels=15,
        out_channels=out_channels,
        clothes_channels=3,
        down_features=(32, 64, 128),
        up_features=(128, 64, 32),
        batch_norm=False,
        instance_norm=True,
    )
    output = model(input_x, input_clothes)
    # plt.imshow(output.detach().numpy()[0, 0, :, :])
    # plt.show()
    # plt.imshow(output.detach().numpy()[0, 1, :, :])
    # plt.show()
    # plt.imshow(output.detach().numpy()[0, 2, :, :])
    # plt.show()
    print(output.shape)
    assert output.shape == (1, out_channels, 256, 192), "Test 1 Failed"
    print("Test 1 Complete")


if __name__ == "__main__":
    test_output_shape()
