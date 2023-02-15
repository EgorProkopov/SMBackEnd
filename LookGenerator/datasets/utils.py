import os
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
