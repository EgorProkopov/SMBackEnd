import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from dataclasses import dataclass
import cv2

@dataclass
class DirInfo:
    name: str
    extension: str


def load_image_for_test(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    # print(root_dir + dir_name + file_name + extension)
    # input()
    # return cv2.imread(root_dir + dir_name + file_name + extension, cv2.IMREAD_COLOR)
    return Image.open(root_dir + dir_name + file_name + extension)


def load_image(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    # print(root_dir + dir_name + file_name + extension)
    # input()
    # return cv2.imread(root_dir + dir_name + file_name + extension, cv2.IMREAD_COLOR)
    return Image.open(  # root_dir + dir_name + file_name + extension)
        os.path.join(
            root_dir,
            dir_name,
            file_name + extension
        )
    )


def convert_channel(image: Image):
    return np.asarray(image.convert('L')) / 255


def clean_image_by_mask(image, mask):
    height, width = image.shape[0], image.shape[1]
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                image[i, j] = [255, 255, 255]


def prepare_image_for_segmentation(image: Image,
                                   transform=None):
    """
    На вход подается трехканальная картинка (высота, ширина, количество каналов)
    Выдает тензор [новое измерение, количество каналов, ширина, высота] вместе с необходимыми преобразованиями
    (при наличии оных)
    """
    tensor = torch.tensor(np.asarray(image, dtype=np.float32).T[np.newaxis, ...])
    tensor = torch.transpose(tensor, 3, 2)
    tensor = transform(tensor)

    return tensor


def to_array_from_model_transpose(tensor):
    """
    На вход подается тензор из модели [измерение, количество каналов, ширина, высота]
    На выход получается массив numpy [высота, ширина, количество каналов]
    """
    return tensor.detach().numpy()[0, :, :, :]


def to_array_from_model_bin(tensor):
    """
    На вход подается тензор из модели [измерение, количество каналов, ширина, высота]
    На выход получается массив numpy [высота, ширина, количество каналов]
    """
    return tensor.detach().numpy()[0, 0, :, :]


def prepare_images_for_encoder(human_image: Image, pose_points_list: list, clothes_image: Image,
                               input_rgb_transform=None, input_bin_transform=None):
    """
    Function for images preparation before encoder-decoder
    Args:
        human_image: a human image for encoder-decoder input
        pose_points_list: a list of images of pose points for encoder-decoder input
        clothes_image: clothes image for encoder-decoder input
        input_rgb_transform: input images transform
        input_bin_transform: input pose points transform

    Returns:
        Torch tensor that can be an input of encoder-decoder model
    """
    to_tensor = transforms.ToTensor()
    human_image = to_tensor(human_image)
    if input_rgb_transform:
        human_image = input_rgb_transform(human_image)

    # Pose points
    pose_points = torch.empty(0)
    for pose_point in pose_points_list:
        pose_point = to_tensor(pose_point)
        if input_bin_transform:
            pose_point = input_bin_transform(pose_point)

        pose_points = torch.cat((pose_points, pose_point))

    # Clothes
    clothes_image = to_tensor(clothes_image)
    if input_rgb_transform:
        clothes_image = input_rgb_transform(clothes_image)

    enc_dec_input = torch.cat((pose_points, human_image, clothes_image), axis=0)
    enc_dec_input = torch.reshape(enc_dec_input, (
        1,
        enc_dec_input.shape[0],
        enc_dec_input.shape[1],
        enc_dec_input.shape[2]
    ))
    return enc_dec_input


def to_array_from_decoder(tensor):
    tensor = torch.transpose(tensor, 3, 1)
    tensor = torch.transpose(tensor, 1, 2)
    return tensor.detach().numpy()[0]


def to_image_from_decoder(tensor):
    array = to_array_from_decoder(tensor)
    im = Image.fromarray(np.uint8(255*array)) .convert('RGB')
    return im


def show_array_as_image(array: np.array):
    return plt.imshow(array)


def save_array_as_image(array: np.array, save_path: str):
    """
    Method to save image from numpy array
    Args:
        array: a source array
        save_path: path to save an image
    """
    image = Image.fromarray(array)
    image.save(save_path)


def show_array_multichannel(array, num_channels):

    """
    На вход подается массив numpy [ВЫСОТА, ШИРИНА, КАНАЛЫ]
    каналов 16!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, иначе пользоваться предыдущим методов для каждого канала отдельно

    Вспомогательная функция, которая была использована для показа 16 канальных изображений
    """
    plt.figure(figsize=(18, 6))
    for i in range(1, num_channels // 2 + 1 + num_channels % 2):
        plt.subplot(1, 8, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(array[i - 1, :, :], cmap='gray')

    plt.figure(figsize=(18, 6))
    ctr = 0
    for i in range(num_channels // 2 + num_channels % 2, num_channels):  # mb need fix
        ctr += 1
        plt.subplot(1, 8, ctr)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(array[i, :, :], cmap='gray')
