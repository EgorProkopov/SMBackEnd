import os
import torch
import torchvision.transforms as transforms

from typing import Tuple
from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image


class PosePackDataset(Dataset):
    """Dataset for Encoder-Decoder model training"""
    def __init__(self, image_dir: str,
                 transform_pose_heatmap=None,
                 transform_human=None,
                 transform_pose_points=None,
                 transform_human_restored=None,
                 transform_clothes_mask=None):

        super().__init__()

        self.root = image_dir
        self.transform_pose_heatmap = transform_pose_heatmap
        self.transform_human = transform_human
        self.transform_pose_points = transform_pose_points
        self.transform_human_restored = transform_human_restored
        self.transform_clothes_mask = transform_clothes_mask

        # TODO: прописать пути до всех папок для датасета
        list_of_pose_heatmap_files = os.listdir(image_dir)
        list_of_human_files = os.listdir(image_dir)
        list_of_pose_points_files = os.listdir(image_dir)
        list_of_human_image_files = os.listdir(image_dir)
        list_of_clothes_mask_files = os.listdir(image_dir)

        self.list_of_pose_heatmap_files = list_of_pose_heatmap_files
        self.list_of_human_files = list_of_human_files
        self.list_of_pose_points_files = list_of_pose_points_files
        self.list_of_human_image_files = list_of_human_image_files
        self.list_of_clothes_mask_files = list_of_clothes_mask_files

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        to_tensor_transform = transforms.ToTensor()

        # TODO: прописать загрузку картинок
        pose_heatmap_image = load_image(self.root, "image", self.list_of_pose_heatmap_files[idx], ".jpg")
        pose_heatmap_image = to_tensor_transform(pose_heatmap_image)
        if self.transform_pose_heatmap:
            pose_heatmap_image = self.transform_pose_heatmap(pose_heatmap_image)

        human_image = load_image(self.root, "image", self.list_of_human_files[idx], ".jpg")
        human_image = to_tensor_transform(human_image)
        if self.transform_human:
            human_image = self.transform_human(human_image)

        # pose_points = ???
        human_restored_image = load_image(self.root, "image", self.list_of_human_image_files[idx], ".jpg")
        human_restored_image = to_tensor_transform(human_restored_image)
        if self.transform_human_restored:
            human_restored_image = self.transform_human_restored(human_restored_image)

        clothes_mask_image = load_image(self.root, "image", self.list_of_clothes_mask_files[idx], ".jpg")
        clothes_mask_image = to_tensor_transform(clothes_mask_image)
        if self.transform_clothes_mask:
            clothes_mask_image = self.transform_clothes_mask(clothes_mask_image)

        input_ = torch.cat((pose_heatmap_image, human_image), axis=0)  # не забыть про pose_points
        target = torch.cat((human_restored_image, clothes_mask_image), axis=0)

        return input_, target

    def __len__(self):
        return len(self.list_of_human_files)
