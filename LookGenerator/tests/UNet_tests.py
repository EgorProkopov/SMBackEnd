import torch

from LookGenerator.networks.segmentation import UNet
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor


def test_output_shape():
    input_value = Image.open(Path("test_data_samples/cloth.jpg"))
    transforms = ToTensor()
    input_value = transforms(input_value)[None, :]

    model = UNet(in_channels=3, out_channels=3)
    output_value = model(input_value)
    output_value = torch.transpose(output_value, 1, 3)
    output_value = torch.transpose(output_value, 1, 2)
    print(output_value.shape)
    assert output_value.shape == (1, 1024, 768, 3)


if __name__ == "__main__":
    test_output_shape()
