import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from dataclasses import dataclass


@dataclass
class DirInfo:
    name: str
    extension: str


def load_image_for_test(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    return Image.open(root_dir + dir_name + file_name + extension)


def load_image(root_dir: str, dir_name: str, file_name: str, extension: str) -> Image:
    return Image.open(  # root_dir + dir_name + file_name + extension)
        os.path.join(
            root_dir,
            dir_name,
            file_name + extension
        )
    )


def clean_image_by_mask(image, mask):
    result = torch.clone(image)
    height, width = image.shape[1], image.shape[2]
    for i in range(height):
        for j in range(width):
            if mask[0, i, j] == 1:
                result[:, i, j] = torch.tensor([1, 1, 1])
    
    return result


def prepare_image_for_segmentation(image: Image,
                                   transform=None):
    """
    На вход подается трехканальная картинка (высота, ширина, количество каналов)
    Выдает тензор [новое измерение, количество каналов, ширина, высота] вместе с необходимыми преобразованиями
    (при наличии оных)
    """
    tensor = transforms.ToTensor()(image)
    tensor = tensor.unsqueeze(0)
    tensor = transform(tensor)

    return tensor


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
