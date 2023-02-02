import os

from LookGenerator.datasets.classes import PersonSegmentationDataset, ClothesSegmentationDataset
from LookGenerator.config.config import DATASET_DIR


def test_person_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = PersonSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = PersonSegmentationDataset(test_path)
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    assert dataset[0]["image"].shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['image'].shape}"
    assert dataset[0]["densepose"].shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['mask'].shape}"
    assert dataset[0]["parse_agnostic"].shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['mask'].shape}"
    assert dataset[0]["parse"].shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['mask'].shape}"


def test_cloth_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = ClothesSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = ClothesSegmentationDataset(test_path)
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    assert dataset[0]["image"].shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['image'].shape}"
    assert dataset[0]["mask"].shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expected (3, 1024, 768), got {dataset[0]['mask'].shape}"


