import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from LookGenerator.networks.encoder_decoder import EncoderDecoder


def test_output_shape():
    input_ = Image.open(Path("test_data_samples/image.jpg"))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 192))
    ])
    input_ = transform(input_)[None, :]

    out_channels = 3

    model = EncoderDecoder(in_channels=3, out_channels=out_channels)
    output = model(input_)
    plt.imshow(output.detach().numpy()[0, 0, :, :])
    plt.show()
    plt.imshow(output.detach().numpy()[0, 1, :, :])
    plt.show()
    plt.imshow(output.detach().numpy()[0, 2, :, :])
    plt.show()
    print(output.shape)
    assert output.shape == (1, out_channels, 256, 192)


if __name__ == "__main__":
    test_output_shape()
