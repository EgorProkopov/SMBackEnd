import PIL.Image as Image
import os

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms



class ShirtsDataset(Dataset):
    def __init__(self, root, transform=None, transform_mask=None):
        super(ShirtsDataset, self).__init__()
        self.root = root
        self.files_list = os.listdir(os.path.join(
            root,
            "cloth"
        ))
        self.transform = transform
        self.transform_mask = transform_mask

    def __getitem__(self, idx):

        totensor = transforms.ToTensor()
        file = self.files_list[idx]

        mask_root = os.path.join(self.root,
                                 "image-parse-v3-multichannel",
                                 file.split('.')[0])

        mask_folder = os.listdir(mask_root)
        mask = torch.tensor([])

        for m in mask_folder:
            mm = Image.open(os.path.join(mask_root, m))
            mm = totensor(mm)
            mask = torch.cat((mask, mm), dim=0)

        shirt = Image.open(os.path.join(
            self.root,
            "cloth",
            file
        ))
        shirt_mask = Image.open(os.path.join(
            self.root,
            "cloth-mask",
            file
        ))
        person = Image.open(os.path.join(
            self.root,
            "image",
            file
        ))

        # pose_points = get_pose_points(os.path.join(
        #     self.root,
        #     "openpose_json",
        #     file.split('.')[0] + "_keypoints.json"
        # ))
        # #pose_points = torch.Tensor(pose_points)

        shirt = totensor(shirt)
        person = totensor(person)
        shirt_mask = totensor(shirt_mask)

        torso = mask[8]
        torso = torso.unsqueeze(0)
        invmask = ~(torso.type('torch.BoolTensor'))

        warped_shirt = person * torso + invmask

        if self.transform:
            shirt = self.transform(shirt)
            warped_shirt = self.transform(warped_shirt)

        if self.transform_mask:
            shirt_mask = self.transform_mask(shirt_mask)
            torso = self.transform_mask(torso)
            mask = self.transform_mask(mask)

        result = {
            'shirt': shirt,
            'warped_shirt': warped_shirt,
            'shirt_mask': shirt_mask,
            'segmentation': mask,
            'torso': torso

        }

        return result

    def __len__(self):
        return len(self.files_list)
