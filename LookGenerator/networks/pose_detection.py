import os
import numpy as np

import cv2
import torch
from torchvision.transforms import ToTensor, Resize

from LookGenerator.config.config import PROJECT_ROOT


def _valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = _valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, scale


def load_posenet():
    """
        Loads posenet model
    """
    path = os.path.join(PROJECT_ROOT, "weights", "posenet.pt")
    net = torch.jit.load(path)
    return net


def process_image(net, image, width=192, height=256):
    """
    Args:
        net: PoseNet
        image: CV2 image
        width: width of the resulting map
        height: height of the resulting map
    Returns tuple of 4 maps:
        headmap: torch.Tensor, shape=(17, width, height)
        offsets: torch.Tensor, shape=(34, width, height)
        displacements_fwd:  torch.Tensor, shape=(34, width, height)
        displacements_bwd:  torch.Tensor, shape=(34, width, height)
    """

    image, scale = _process_input(image)

    to_tensor = ToTensor()

    image = to_tensor(image)
    heatmap, offsets, displacements_fwd, displacements_bwd = net(image)
    resize = Resize((height, width))

    result = (heatmap, offsets, displacements_fwd, displacements_bwd)
    result = (x.squeeze(0) for x in result)
    result = (resize.forward(x) for x in result)

    return result
    