import os
from collections.abc import Callable
from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from kornia.constants import DataKey, Resample
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.datasets.cdl import CDL
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.nlcd import NLCD
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential


class EnMAPCDLNLCDBenchmark(NonGeoDataset):
    """EnMAP-CDL/NLCD dataset class."""

    valid_products = ["cdl", "nlcd"]
    valid_splits = ["train", "val", "test"]

    image_root = "{}"
    mask_root = "{}"
    split_path = "splits/{}_{}/{}.txt"

    rgb_indices = {"enmap": [43, 28, 10]}

    split_percentages = [0.75, 0.1, 0.15]

    cmaps = {"nlcd": NLCD.cmap, "cdl": CDL.cmap}

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        product: str = "cdl",
        split: str = "train",
        classes: list[int] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        num_bands: int = 202,
        raw_mask: bool = False,
    ) -> None:
        """Initialize the EnMAP-CDL/NLCD dataset.
        Args:
            root: root directory where dataset can be found
            sensor: ['enmap']
            product: mask target, one of ['cdl', 'nlcd']
            split: dataset split, one of ['train', 'val', 'test']
            classes: list of classes to include, the rest will be mapped to n_classes
                (defaults to all classes for the chosen product)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if any arguments are invalid
        """

        self.sensor = sensor
        assert product in self.valid_products, (
            f"Only supports one of {self.valid_products}, but found {product}."
        )
        self.product = product
        assert split in self.valid_splits, (
            f"Only supports one of {self.valid_splits}, but found {split}."
        )
        self.split = split

        self.cmap = self.cmaps[product]
        if classes is None:
            classes = list(self.cmap.keys())

        assert 0 in classes, "Classes must include the background class: 0"

        self.root = root
        self.classes = classes
        self.transforms = transforms
        self.ordinal_map = (
            torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.long)
            + len(self.classes)
            - 1
        )
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.img_dir_name = self.image_root.format(self.sensor)
        self.mask_dir_name = self.mask_root.format(self.product)
        self.num_bands = num_bands
        self.raw_mask = raw_mask

        # Check if split file exists, if not create it
        self.split_file = self.split_path.format(self.sensor, self.product, self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

        # First remove class 0 from the list of classes
        self.classes.remove(0)
        # Then add it back to the end of the list. This ensures that the background class is always the last class, which can be ignored during training.
        self.classes.append(0)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def split_train_val_test(self) -> list:
        """Random Split Train/Val/Test. Not used in the current implementation. The function was used to generate the split files."""
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

        return train_val_test_images

    def read_split_file(self) -> list:
        """Read .txt file containing train/val/test split. Every row has the following format: path/to/image path/to/mask"""
        with open(self.split_file) as f:
            sample_collection = [x.strip().split(" ") for x in f.readlines()]
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        img_path, mask_path = self.sample_collection[index]

        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)

    def _load_image(self, path: str) -> Tensor:
        """Load the input image.

        Args:
            path: path to input image

        Returns:
            image
        """
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read()).float()
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        if self.raw_mask:
            return mask
        return self.ordinal_map[mask]

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
        ncols = 2

        image = sample["image"][self.rgb_indices[self.sensor]].numpy()

        # image = equalize_image(image) # Optional, only for visualization purposes

        image = image.transpose(1, 2, 0)

        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class EnMAPCDLNLCDBenchmarkDataModule(NonGeoDataModule):
    """LightningDataModule for the EnMAP-CDL/NLCD dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 128,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new EnMAP-CDL/NLCD instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`EnMAPCDLNLCDBenchmark`.
        """
        super().__init__(EnMAPCDLNLCDBenchmark, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.aug = None

        sensor = kwargs.get("sensor")

        try:
            mean = torch.tensor(np.load("statistics/mu.npy"))
            std = torch.tensor(np.load("statistics/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException(f"Missing statistics for sensor {sensor}")

        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.4, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(mean=mean, std=std),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            K.Normalize(mean=mean, std=std),
            data_keys=["image", "mask"],
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing:
                aug = self.test_aug
            elif self.trainer.predicting:
                aug = self.test_aug
            else:
                print("No trainer mode found")
                raise NotImplementedError
            batch["image"] = batch["image"].float()
            batch = aug(batch)

        return batch

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'train_dataset'.
        """
        dataset = self.train_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.train_batch_size or self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                persistent_workers=True,
            )
        else:
            raise NotImplementedError("No dataset found")

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'val_dataset'.
        """
        dataset = self.val_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.val_batch_size or self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True,
                # drop_last=True,
                persistent_workers=True,
            )
        else:
            raise NotImplementedError("No dataset found")


class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        super().__init__()

        mean = torch.tensor(np.load("statistics/mu.npy"))
        std = torch.tensor(np.load("statistics/sigma.npy"))

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Resize(_to_tuple(size)),
                K.RandomResizedCrop(_to_tuple(size), scale=(0.4, 1.0)),
                K.RandomVerticalFlip(p=0.5),
                K.RandomHorizontalFlip(p=0.5),
                K.Normalize(mean=mean, std=std),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.Resize(_to_tuple(size)),
                K.CenterCrop(size),
                K.Normalize(mean=mean, std=std),
            )

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"]


class EnMAPCDLNLCDDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size)

        dataset_train = EnMAPCDLNLCDBenchmark(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = EnMAPCDLNLCDBenchmark(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = EnMAPCDLNLCDBenchmark(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
