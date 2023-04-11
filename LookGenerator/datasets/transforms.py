import torch
import torchvision.transforms as transforms


class MinMaxScale(object):
    """
    Min-Max scale transform to [0, 1]

    data_transformed = (data -  data_min) / (data_max - data_min)
    """
    def __init__(self):
        self.min = 0
        self.max = 0

    def __call__(self, image):
        self.min = image.min()
        self.max = image.max()

        min_tensor = torch.full(image.shape, self.min)
        if (self.max - self.min) == 0:
            self.max = 1e-10
            self.min = 0

        image = (image - min_tensor) / (self.max - self.min)
        return image


class ThresholdTransform(object):
    """Signum function transform"""
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold: threshold to put 1 or 0. Value can be between 0 and 1
        """
        self.threshold = threshold

    def __call__(self, image):
        """
        Args:
            image: image to be transformed
        """
        return (image >= self.threshold).float()


class Normalize(object):
    """OUR normalization"""
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, tensor: torch.Tensor):
        batched = True if len(tensor.size()) == 4 else False

        if not batched:
            self.mean = [tensor[0].mean(), tensor[1].mean(), tensor[2].mean()]
            self.std = [tensor[0].std(), tensor[1].std(), tensor[2].std()]

        else:
            self.mean = [tensor[0,0].mean(), tensor[0, 1].mean(), tensor[0, 2].mean()]
            self.std = [tensor[0, 0].std(), tensor[0, 1].std(), tensor[0, 2].std()]

        normalize = transforms.Normalize(mean=self.mean,
                                         std=self.std)

        tensor = normalize(tensor)
        

        return tensor


class DividerScaler(object):
    """
    Scalar division of tensor
    """
    def __init__(self, div):
        self.div = div

    def __call__(self, image):
        return image / self.div
