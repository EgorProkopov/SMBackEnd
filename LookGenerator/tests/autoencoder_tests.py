import torch
import torch.nn as nn

from LookGenerator.networks.clothes_feature_extractor import ClothAutoencoder


def shape_test(input_tensor):
    autoencoder = ClothAutoencoder(
        in_channels=3,
        out_channels=3,
        features=(8, 16, 32, 64),
        latent_dim_size=128,
        encoder_activation_func=nn.LeakyReLU(),
        decoder_activation_func=nn.ReLU()
    )

    input_tensor_shape = input_tensor.shape
    output_tensor = autoencoder(input_tensor)
    output_tensor_shape = output_tensor.shape

    assert input_tensor_shape == output_tensor_shape, "Test 1 Failed"

    print("Test 1 Complete")


def str_test():
    autoencoder = ClothAutoencoder(
        in_channels=3,
        out_channels=3,
        features=(8, 16, 32, 64),
        latent_dim_size=128,
        encoder_activation_func=nn.LeakyReLU(),
        decoder_activation_func=nn.ReLU()
    )

    print("Test 2 Result: ")
    print(str(autoencoder))


if __name__ == "__main__":
    input_tensor = torch.zeros((1, 3, 256, 192))
    shape_test(input_tensor)
    str_test()