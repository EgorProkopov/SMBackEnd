import torch


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))
    return model
