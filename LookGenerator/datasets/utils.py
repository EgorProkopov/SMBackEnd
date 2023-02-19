import os

import matplotlib.pyplot as plt
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


def prepare_image_for_model(image: Image):
    return np.asarray(image, dtype=np.float32)


def show_array_as_image(array: np.array):
    return plt.imshow(array)

