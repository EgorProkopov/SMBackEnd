import torch


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))
    return model    


def get_num_digits(a):
    num = 0
    while a > 0:
        num += 1
        a = a // 10
    return num
