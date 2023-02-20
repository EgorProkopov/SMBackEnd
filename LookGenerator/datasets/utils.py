import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class DirInfo:
    name: str
    extension: str


def load_image(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    return Image.open(
        os.path.join(
            root_dir,
            dir_name,
            file_name + extension
        )
    )


def convert_channel(image: Image):
    return np.asarray(image.convert('L')) / 255


def prepare_image_for_model_transpose(image: Image):
    """
    На вход подается трехканальная картинка (высота, ширина, количество каналов)
    Выдает тензор [новое измерение, количество каналов, ширина, высота]
    """
    return torch.tensor(np.asarray(image, dtype=np.float32)[..., np.newaxis].T)


def prepare_image_for_model(image: Image):
    """
    На вход подается трехканальная картинка (количество каналов, ширина, высота)
    На выход подается трехканальная картинка (новое измерение, количество каналов, ширина, высота)
    """
    return torch.tensor(np.asarray(image, dtype=np.float32)[np.newaxis, ...])


def to_array_from_model_transpose(tensor):
    """
    На вход подается тензор из модели [измерение, количество каналов, ширина, высота]
    На выход получается массив numpy [высота, ширина, количество каналов]
    """
    return tensor.detach().numpy()[0, :, :, :].T


def show_array_as_image(array: np.array):
    return plt.imshow(array)


def show_array_multichannel(array):

    """
    На вход подается массив numpy [ВЫСОТА, ШИРИНА, КАНАЛЫ]
    каналов 16!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, иначе пользоваться предыдущим методов для каждого канала отдельно

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
    for i in range(8, 16):
        ctr += 1
        plt.subplot(1, 8, ctr)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(array[:, :, i], cmap='gray')
