from LookGenerator.networks.segmentation import UNet
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor


def test_output_shape():
    input_value = Image.open(Path("test_data_samples/cloth.jpg"))
    transforms = ToTensor()
    input_value = transforms(input_value)[None, :]

    model = UNet()
    output_value = model(input_value)
    print(output_value.shape)
    assert output_value.shape == (1, 1, 1024, 768)


if __name__ == "__main__":
    test_output_shape()
