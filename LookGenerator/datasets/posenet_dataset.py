from torch.utils.data import Dataset


class PoseNetDataset(Dataset):
    """
        Dataset for Pose detection
    """

    def __init__(self, image_dir: str, transform=None):
        self.root = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        return 0
