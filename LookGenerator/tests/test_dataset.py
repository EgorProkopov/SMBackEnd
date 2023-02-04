import os

from LookGenerator.datasets.classes import PersonSegmentationDataset, ClothesSegmentationDataset
from LookGenerator.config.config import DATASET_DIR


def test_person_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = PersonSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = PersonSegmentationDataset(test_path, segmentation_type="densepose")
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    input_, target = dataset[0]

    assert input_.shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {input_.shape}"
    assert target.shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {target.shape}"

    dataset = PersonSegmentationDataset(test_path, segmentation_type="parse")
    _, target = dataset[0]
    assert target.shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (1, 1024, 768), got {target.shape}"

    dataset = PersonSegmentationDataset(test_path, segmentation_type="parse-agnostic")
    _, target = dataset[0]
    assert target.shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (1, 1024, 768), got {target.shape}"


def test_cloth_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = ClothesSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = ClothesSegmentationDataset(test_path)
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    input_, targets = dataset[0]
    assert input_.shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {input_.shape}"
    assert targets.shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {targets.shape}"
