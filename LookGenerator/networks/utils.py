import os
import numpy as np

import torch


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str):
    model = torch.load(path)
    return model
