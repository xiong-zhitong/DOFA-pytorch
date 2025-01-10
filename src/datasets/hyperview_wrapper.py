import glob
import os
from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential


def resize_hyperspectral_tensor(img_tensor, output_shape=(150, 128, 128)):
    img_tensor = img_tensor.unsqueeze(0)

    resized_tensor = F.interpolate(
        img_tensor, size=output_shape[1:], mode="bicubic", align_corners=True
    )

    resized_tensor = resized_tensor.squeeze(0)

    return resized_tensor


class HyperviewBenchmark(NonGeoDataset):
    valid_splits = ["train", "val", "test"]

    split_path = "splits/hyperview/{}.txt"
    gt_file_path = "gt_{}.csv"

    rgb_indices = [43, 28, 10]

    split_percentages = [0.75, 0.1, 0.15]

    def __init__(
        self,
        root: str = "data",
        data_dir: str = "images",
        split: str = "train",
        num_bands: int = 150,
        patch_size: int = 128,
    ) -> None:
        assert split in self.valid_splits, (
            f"Only supports one of {self.valid_splits}, but found {split}."
        )
        self.split = split

        self.root = root
        self.img_path = os.path.join(self.root, data_dir)
        self.gt_path = os.path.join(self.root, self.gt_file_path.format(self.split))
        self.num_bands = num_bands
        self.patch_size = patch_size

        # Check if split file exists, if not create it
        self.split_file = self.split_path.format(self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
            self.gt = self.load_gt(os.path.join(self.root, self.gt_path))
        else:
            raise ValueError("Split file does not exist. Please create it first.")

        # Read all images
        self.images = [self.read_image(img_path) for img_path in self.sample_collection]
        self.labels = self.gt

    def split_train_val_test(self) -> list:
        """Split Train/Val/Test at the tile level."""
        np.random.seed(0)
        sizes = (np.array(self.split_percentages) * len(self.sample_collection)).astype(
            int
        )
        cutoffs = np.cumsum(sizes)[:-1]
        sample_indices = np.arange(len(self.sample_collection))
        np.random.shuffle(sample_indices)
        groups = np.split(sample_indices, cutoffs)
        split_indices = {"train": groups[0], "val": groups[1], "test": groups[2]}

        train_val_test_images = {
            "train": [self.sample_collection[idx] for idx in split_indices["train"]],
            "val": [self.sample_collection[idx] for idx in split_indices["val"]],
            "test": [self.sample_collection[idx] for idx in split_indices["test"]],
        }
        train_val_test_gt = {
            "train": self.gt.iloc[split_indices["train"]],
            "val": self.gt.iloc[split_indices["val"]],
            "test": self.gt.iloc[split_indices["test"]],
        }

        return train_val_test_images, train_val_test_gt

    def read_split_file(self) -> list:
        """Read .txt file containing train/val/test split. Every row has the following format: path/to/image path/to/mask"""
        sample_collection = []

        with open(self.split_file) as f:
            for line in f:
                img_path = line.strip()
                sample_collection.append(img_path)

        return sample_collection

    def read_image(self, img_path: str) -> np.ndarray:
        """Read image from .npz file."""
        with np.load(img_path) as npz:
            arr = np.ma.MaskedArray(**npz).data
        return arr

    def load_gt(self, file_path: str):
        """Load labels for train set from the ground truth file.
        Args:
            file_path (str): Path to the ground truth .csv file.
        Returns:
            [type]: 2D numpy array with soil properties levels
        """
        gt_file = pd.read_csv(file_path)
        labels = gt_file[["P", "K", "Mg", "pH"]].values
        return labels

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        image = torch.tensor(self.images[index]).float()
        image = resize_hyperspectral_tensor(
            image, output_shape=(self.num_bands, self.patch_size, self.patch_size)
        )

        sample = {"image": image, "label": self.labels[index]}

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)

    def retrieve_sample_collection(self) -> list[tuple[str, str]]:
        """Retrieve paths to samples in data directory."""

        sample_collection = sorted(
            glob.glob(os.path.join(self.img_path, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )

        return sample_collection

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        pass


class HyperviewBenchmarkDataModule(NonGeoDataModule):
    """Lightning DataModule for the Hyperview dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 128,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(HyperviewBenchmark, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.aug = None

        # Replace this with yours
        mean = torch.tensor(np.load("data/hyperview_statistics/mu.npy"))
        std = torch.tensor(np.load("data/hyperview_statistics/sigma.npy"))

        self.train_aug = AugmentationSequential(
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(mean=mean, std=std),
            data_keys=["image"],
        )
        self.val_aug = AugmentationSequential(
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=["image"],
        )
        self.test_aug = AugmentationSequential(
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=["image"],
        )


class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        super().__init__()

        mean = torch.tensor(np.load("data/hyperview_statistics/mu.npy"))
        std = torch.tensor(np.load("data/hyperview_statistics/sigma.npy"))

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.RandomResizedCrop(_to_tuple(size), scale=(0.6, 1.0)),
                K.RandomVerticalFlip(p=0.5),
                K.RandomHorizontalFlip(p=0.5),
                K.Normalize(mean=mean, std=std),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.CenterCrop(size), K.Normalize(mean=mean, std=std)
            )

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"]


class HyperviewDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size)

        dataset_train = HyperviewBenchmark(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = HyperviewBenchmark(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = HyperviewBenchmark(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
