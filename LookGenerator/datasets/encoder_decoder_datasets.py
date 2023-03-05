import os
import torch
import torchvision.transforms as transforms

from typing import Tuple
from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image, convert_channel


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

        list_of_human_no_clothes_files = os.listdir(os.path.join(image_dir, "imageWithNoCloth"))
        list_of_pose_points_files = os.listdir(os.path.join(image_dir, "posePoints"))
        list_of_clothes_files = os.listdir(os.path.join(image_dir, "cloth"))
        list_of_human_image_files = os.listdir(os.path.join(image_dir, "image"))

        self.list_of_human_no_clothes_files = list_of_human_no_clothes_files
        self.list_of_pose_points_files = list_of_pose_points_files
        self.list_of_clothes_files = list_of_clothes_files
        self.list_of_human_image_files = list_of_human_image_files

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        to_tensor_transform = transforms.ToTensor()
        to_tensor = torch.ToTensor()

        # Human image
        human_image = load_image(self.root, "imageWithNoCloth", self.list_of_human_no_clothes_files[idx], ".png")
        human_image = to_tensor_transform(human_image)

        if self.transform_human:
            human_image = self.transform_human(human_image)

        # Pose points
        # Num of pose points: 17
        pose_points = torch.empty(0)

        points_list = os.listdir(os.path.join(
            self.root,
            "posePoints",
            self.list_of_pose_points_files[idx]
        ))

        pose_points_list = [file.split('.')[0] for file in points_list]

        for pose_point in pose_points_list:
            point = convert_channel(load_image(
                                                self.root,
                                                os.path.join("posePoints", self.list_of_pose_points_files[idx]),
                                                pose_point,
                                                ".png"))
            point = to_tensor(point)

            if self.transform_pose_points:
                point = self.transform_pose_points(point)

            pose_points.cat((pose_points, point), axis=0)


        # Clothes
        clothes_image = load_image(self.root, "cloth", self.list_of_clothes_files[idx], ".jpg")
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
        return len(self.list_of_human_no_clothes_files)
