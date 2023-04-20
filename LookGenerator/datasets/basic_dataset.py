from torchvision.transforms import ToTensor
import os
from torch.utils.data import Dataset
from LookGenerator.datasets.utils import load_image


class BasicDataset(Dataset):
    """ Basic dataset with transforms for images and targets"""

    def __init__(self,
                 root_dir: str,
                 input_dir_name: str,
                 target_dir_name: str,
                 transform_input=None,
                 transform_target=None):
        """

        Args:
            root_dir: dir where a folder is
            input_dir_name: folder with inputs
            target_dir_name: folder with targets
            transform_input: transform for input
            transform_target: transform for target
        """
        super().__init__()

        self.root_dir = root_dir
        self.input_dir_name = input_dir_name
        self.target_dir_name = target_dir_name
        self.transform_input = transform_input
        self.transform_target = transform_target

        input_root = os.path.join(root_dir, input_dir_name)
        target_root = os.path.join(root_dir, target_dir_name)

        input_files = os.listdir(input_root)
        target_files = os.listdir(target_root)

        self._input_files_list = [name.split('.')[0] for name in input_files]
        self._input_extensions_list = [name.split('.')[1] for name in input_files]

        self._target_files_list = [name.split('.')[0] for name in target_files]
        self._target_extensions_list = [name.split('.')[1] for name in target_files]

    def __getitem__(self, idx):
        to_tensor = ToTensor()

        input_ = load_image(self.root_dir,
                            self.input_dir_name,
                            self._input_files_list[idx],
                            '.' + self._input_extensions_list[idx])
        target = load_image(self.root_dir,
                            self.target_dir_name,
                            self._target_files_list[idx],
                            '.' + self._target_extensions_list[idx])
        input_ = to_tensor(input_)
        target = to_tensor(target)

        if self.transform_input:
            input_ = self.transform_input(input_)

        if self.transform_target:
            target = self.transform_target(target)

        return input_, target

    def __len__(self):
        return len(self._input_files_list)
