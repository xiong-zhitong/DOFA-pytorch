"""Dummy Dataset that can be used for testing implementations."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import kornia.augmentation as K
from dataclasses import dataclass
from typing import Tuple, Dict, Any


class DummyDataAugmentation(torch.nn.Module):
    """Augmentation pipeline matching real dataset interface."""

    def __init__(self, split: str, size: Tuple[int, int]) -> None:
        super().__init__()
        mean = torch.Tensor([0.0])
        std = torch.Tensor([1.0])

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            )

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"]


class DummyDataset(Dataset):
    """Dummy dataset generating random samples."""

    def __init__(
        self,
        img_size: int,
        num_channels: int,
        num_classes: int,
        split: str,
        transforms: Any,
        task: str = "classification",
        num_samples: int = 2,
    ) -> None:
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = num_channels
        self.task = task

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random image and label
        image = torch.rand(self.channels, *self.img_size)
        if self.task == "classification":
            label = torch.randint(0, self.num_classes, (1,)).squeeze(0)
        else:
            label = torch.randint(0, self.num_classes, (1, *self.img_size)).squeeze(0)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class DummyWrapper:
    """Wrapper class matching interface."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.num_classes = config.num_classes
        self.num_channels = config.num_channels
        self.task = config.task

    def create_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Create train, validation and test datasets."""
        train_transform = DummyDataAugmentation(split="train", size=self.img_size)
        eval_transform = DummyDataAugmentation(split="test", size=self.img_size)

        dataset_train = DummyDataset(
            split="train",
            transforms=train_transform,
            img_size=self.img_size,
            num_samples=2,
            task=self.task,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
        )
        dataset_val = DummyDataset(
            split="val",
            transforms=eval_transform,
            img_size=self.img_size,
            num_samples=1,
            task=self.task,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
        )
        dataset_test = DummyDataset(
            split="test",
            transforms=eval_transform,
            img_size=self.img_size,
            num_samples=1,
            task=self.task,
            num_classes=self.num_classes,
            num_channels=self.num_channels,
        )

        return dataset_train, dataset_val, dataset_test
