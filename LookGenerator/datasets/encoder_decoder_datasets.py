import os
import torch
import torchvision.transforms as transforms

from typing import Tuple
from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image


class EncoderDecoderDataset(Dataset):
    """Dataset for Encoder-Decoder model training"""
    def __init__(self, image_dir: str,
                 transform_human=None,
                 transform_pose_points=None,
                 transform_clothes=None,
                 transform_human_restored=None):

        super().__init__()

        self.root = image_dir

        self.transform_human = transform_human
        self.transform_pose_points = transform_pose_points
        self.transform_clothes = transform_clothes
        self.transform_human_restored = transform_human_restored

        # TODO: прописать пути до всех папок для датасета
        list_of_human_files = os.listdir(image_dir)
        list_of_pose_points_files = os.listdir(image_dir)
        list_of_clothes_files = os.listdir(image_dir)
        list_of_human_image_files = os.listdir(image_dir)

        self.list_of_human_files = list_of_human_files
        self.list_of_pose_points_files = list_of_pose_points_files
        self.list_of_clothes_files = list_of_clothes_files
        self.list_of_human_image_files = list_of_human_image_files

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        to_tensor_transform = transforms.ToTensor()

        # TODO: прописать загрузку картинок
        # Human image
        human_image = load_image(self.root, "image", self.list_of_human_files[idx], ".jpg")
        human_image = to_tensor_transform(human_image)
        if self.transform_human:
            human_image = self.transform_human(human_image)

        # Pose points
        # Num of pose points: 17
        pose_points = []

        # Clothes
        clothes_image = load_image(self.root, "image", self.list_of_clothes_files[idx], ".jpg")
        # TODO: проверить расширение файла
        clothes_image = to_tensor_transform(clothes_image)
        if self.transform_clothes:
            clothes_image = self.transform_clothes(clothes_image)

        # Restored image
        human_restored_image = load_image(self.root, "image", self.list_of_human_image_files[idx], ".jpg")
        human_restored_image = to_tensor_transform(human_restored_image)
        if self.transform_human_restored:
            human_restored_image = self.transform_human_restored(human_restored_image)

        input_ = torch.cat((pose_points, human_image, clothes_image), axis=0)  # не забыть про pose_points
        target = human_restored_image

        return input_, target

    def __len__(self):
        return len(self.list_of_human_files)
