import os
import sys

from LookGenerator.datasets.classes import PersonSegmentationDataset, ClothesSegmentationDataset
from LookGenerator.config.config import DATASET_DIR
from pathlib import Path


def test_person_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = PersonSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = PersonSegmentationDataset(test_path)
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    assert dataset[0].shape == (4, 3, 1024, 768)


def test_cloth_segmentation():
    train_path = os.path.join(DATASET_DIR, "train")
    test_path = os.path.join(DATASET_DIR, "test")

    dataset = ClothesSegmentationDataset(train_path)
    assert len(dataset) != 0, "Train dataset is empty or Dataset does not load any."

    dataset = ClothesSegmentationDataset(test_path)
    assert len(dataset) != 0, "Test dataset is empty or Dataset does not load any."

    print(dataset[0])
    assert dataset[0].image.shape == (3, 1024, 768), \
        f"Wrong tensor.shape. Expecter (3, 1024, 768), got {dataset[0].image.shape}"
    assert dataset[0].mask.shape == (1, 1024, 768), \
        f"Wrong tensor.shape. Expecter (3, 1024, 768), got {dataset[0].mask.shape}"


