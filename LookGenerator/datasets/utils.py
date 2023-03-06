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


def prepare_image_for_model(image: Image, transform= None):
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
    return tensor.detach().numpy()[0, :, :, :].T


def to_array_from_model_bin(tensor):
    """
    На вход подается тензор из модели [измерение, количество каналов, ширина, высота]
    На выход получается массив numpy [высота, ширина, количество каналов]
    """
    return tensor.detach().numpy()[0, 0, :, :]


def to_array_from_decoder(tensor):
    tensor = torch.transpose(tensor, 3, 1)
    return tensor.detach().numpy()[0]


def show_array_as_image(array: np.array):
    return plt.imshow(array)


def show_array_multichannel(array):

    """
    На вход подается массив numpy [ВЫСОТА, ШИРИНА, КАНАЛЫ]
    каналов 16!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, иначе пользоваться предыдущим методов для каждого канала отдельно

    Вспомогательная функция, которая была использована для показа 16 канальных изображений
    """

    plt.figure(figsize=(18, 6))
    for i in range(1, 16 // 2 + 1):
        plt.subplot(1, 8, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(array[:, :, i - 1], cmap='gray')

    plt.figure(figsize=(18, 6))
    ctr = 0
    for i in range(8, 15):
        ctr += 1
        plt.subplot(1, 8, ctr)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(array[:, :, i], cmap='gray')
