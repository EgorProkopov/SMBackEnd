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

