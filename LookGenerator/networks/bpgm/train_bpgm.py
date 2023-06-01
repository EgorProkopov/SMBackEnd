#coding=utf-8
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from LookGenerator.networks.bpgm.utils.losses import VGGLoss
from tqdm import tqdm


def train_bpgm(dataloader, model, device='cpu', epochs=1):
    model = model.to(device)
    # criterion
    criterionL1 = nn.L1Loss().to(device)
    criterionVGG = VGGLoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = []
    for epoch in range(epochs):
        epoch_loss = []
        for data in tqdm(dataloader):

            for key in data:
                data[key] = data[key].to(device)

            shirt = data['shirt'].to(device)
            shirt_mask = data['shirt_mask'].to(device)
            warped_shirt = data['warped_shirt'].to(device)
            cwm = data['torso'].to(device)
            mask = data['segmentation'].to(device)

            grid = model(mask, shirt)

            warped = F.grid_sample(shirt, grid, padding_mode='border', align_corners=True)
            warped = warped * cwm
            warped_mask = F.grid_sample(shirt_mask, grid, padding_mode='border', align_corners=True)

            loss_cloth = criterionL1(warped, warped_shirt) + 0.1 * criterionVGG(warped, warped_shirt)
            loss_mask = criterionL1(warped_mask, cwm) * 0.1
            loss = loss_cloth + loss_mask

            optimizer.zero_grad()
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
        losses.append(np.mean(epoch_loss))
        print(losses[:-1])
    return losses