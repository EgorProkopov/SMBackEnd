import os
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image


class EncoderDecoderDataset(Dataset):
    """Dataset for Encoder-Decoder model training"""
    def __init__(self, image_dir: str,
                 pose_points=False,
                 transform_human=None,
                 transform_pose_points=None,
                 transform_clothes=None,
                 transform_human_restored=None,
                 augment=None):
        """

        Args:
            image_dir: dir to the dataset folder
                Directory must contain such subdirectories as:
                    - human
                    - human after segmentation part
                    - pose-points dataset (if it doesn't, don't set pose_point=True)
                    - clothes
            pose_points: if use pose points in encoder-decoder model, set 'True'
            transform_human: transform to be performed on an input human (after segmentation), from pytorch
            transform_pose_points: transform to be performed on an input pose points, from pytorch
            transform_clothes: transform to be performed on an input clothes, from pytorch
            transform_human_restored: transform to be performed on an output restored human, from pytorch
        """
        super().__init__()

        self.pose_points = pose_points

        self.root = image_dir

        self.transform_human = transform_human
        if self.pose_points:
            self.transform_pose_points = transform_pose_points
        self.transform_clothes = transform_clothes
        self.transform_human_restored = transform_human_restored
        self.augment = augment

        list_of_human_no_clothes_files = os.listdir(os.path.join(image_dir, "imageWithNoCloth"))
        if self.pose_points:
            list_of_pose_points_files = os.listdir(os.path.join(image_dir, "posePoints"))
        list_of_clothes_files = os.listdir(os.path.join(image_dir, "cloth"))
        list_of_human_image_files = os.listdir(os.path.join(image_dir, "image"))

        self.list_of_human_no_clothes_files = list_of_human_no_clothes_files
        if self.pose_points:
            self.list_of_pose_points_files = list_of_pose_points_files
        self.list_of_clothes_files = list_of_clothes_files
        self.list_of_human_image_files = list_of_human_image_files

    def __getitem__(self, idx):
        """
        Args:
            idx: The index of data sample

        Returns: A pait of input and target objects

        """
        seed = torch.random.seed()
        to_tensor_transform = transforms.ToTensor()

        # Human image
        human_image = load_image(self.root, "imageWithNoCloth", self.list_of_human_no_clothes_files[idx], "")
        human_image = to_tensor_transform(human_image)

        if self.transform_human:
            torch.manual_seed(seed)
            human_image = self.transform_human(human_image)

        # Pose points
        if self.pose_points:
            # Num of pose points: 17
            pose_points = torch.empty(0)

            points_list = os.listdir(os.path.join(
                self.root,
                "posePoints",
                self.list_of_pose_points_files[idx]
            ))

            pose_points_list = [file.split('.')[0] for file in points_list]

            for pose_point in pose_points_list:
                point = load_image(
                    self.root,
                    os.path.join("posePoints", self.list_of_pose_points_files[idx]),
                    pose_point,
                    ".png").convert('L')
                point = to_tensor_transform(point)

                if self.transform_pose_points:
                    point = self.transform_pose_points(point)

                pose_points = torch.cat((pose_points, point), axis=0)

        # Clothes
        clothes_image = load_image(self.root, "cloth", self.list_of_clothes_files[idx], "")
        clothes_image = to_tensor_transform(clothes_image)

        if self.transform_clothes:
            torch.manual_seed(seed)
            clothes_image = self.transform_clothes(clothes_image)

        # Restored image
        human_restored_image = load_image(self.root, "image", self.list_of_human_image_files[idx], "")
        human_restored_image = to_tensor_transform(human_restored_image)

        if self.transform_human_restored:
            torch.manual_seed(seed)
            human_restored_image = self.transform_human_restored(human_restored_image)

        if self.pose_points:
            input_ = torch.cat((pose_points, human_image, clothes_image), axis=0)
        else:
            input_ = torch.cat((human_image, clothes_image), axis=0)
        target = human_restored_image

        return input_.float(), target.float()
        # return human_image, clothes_image, human_restored_image

    def __len__(self):
        return len(self.list_of_human_no_clothes_files)
