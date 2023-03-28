import os


def check_path_and_creat(path):
    if os.path.isdir(path):
        return True
    if not os.path.isdir(path.rsplit('\\', 1)[0]):
        raise RuntimeError
    os.mkdir(path)
    return True


