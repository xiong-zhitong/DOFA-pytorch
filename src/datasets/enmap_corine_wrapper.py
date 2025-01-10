import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential


class EnMAPCorineBenchmark(NonGeoDataset):
    """

    EnMAP CORINE Dataset.

    """

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    corine_to_label_dict = {
        111: 0,
        112: 1,
        121: 2,
        122: 3,
        123: 4,
        124: 5,
        131: 6,
        132: 7,
        133: 8,
        141: 9,
        142: 10,
        211: 11,
        212: 12,
        213: 13,
        221: 14,
        222: 15,
        223: 16,
        231: 17,
        241: 18,
        242: 19,
        243: 20,
        244: 21,
        311: 22,
        312: 23,
        313: 24,
        321: 25,
        322: 26,
        323: 27,
        324: 28,
        331: 29,
        332: 30,
        333: 31,
        334: 32,
        411: 33,
        412: 34,
        421: 35,
        422: 36,
        423: 37,
        511: 38,
        512: 39,
        521: 40,
        522: 41,
        523: 42,
    }

    label_converter_dict = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    cmap = {
        111: (230, 0, 77),
        112: (255, 0, 0),
        121: (204, 77, 242),
        122: (204, 0, 0),
        123: (230, 204, 204),
        124: (230, 204, 230),
        131: (166, 0, 204),
        132: (166, 77, 0),
        133: (255, 77, 255),
        141: (255, 166, 255),
        142: (255, 230, 255),
        211: (255, 255, 168),
        212: (255, 255, 0),
        213: (230, 230, 0),
        221: (230, 128, 0),
        222: (242, 166, 77),
        223: (230, 166, 0),
        231: (230, 230, 77),
        241: (255, 230, 166),
        242: (255, 230, 77),
        243: (230, 204, 77),
        244: (242, 204, 166),
        311: (128, 255, 0),
        312: (0, 166, 0),
        313: (77, 255, 0),
        321: (204, 242, 77),
        322: (166, 255, 128),
        323: (166, 230, 77),
        324: (166, 242, 0),
        331: (230, 230, 230),
        332: (204, 204, 204),
        333: (204, 255, 204),
        334: (0, 0, 0),
        335: (166, 230, 204),
        411: (166, 166, 255),
        412: (77, 77, 255),
        421: (204, 204, 255),
        422: (230, 230, 255),
        423: (166, 166, 230),
        511: (0, 204, 242),
        512: (128, 242, 230),
        521: (0, 255, 166),
        522: (166, 255, 230),
        523: (230, 242, 255),
        999: (230, 242, 255),
        990: (230, 242, 255),
        995: (230, 242, 255),
    }

    # Extend label converters to classes that don't change

    # Create default dict with 43 as default value for corine_to_label and label_converter
    corine_to_label = defaultdict(lambda: 43, corine_to_label_dict)
    label_converter = defaultdict(lambda: 43, label_converter_dict)

    # url = "https://huggingface.co/datasets/torchgeo/{}/resolve/main/{}.tar.gz"

    valid_sensors = ["enmap"]
    valid_products = ["corine"]
    valid_splits = ["train", "val", "test"]

    image_root = "{}"
    mask_root = "{}"
    split_path = "splits/{}_{}/{}.txt"

    rgb_indices = {
        "enmap": [43, 28, 10],
        "enmap_vnir": [2, 1, 0],
        "enmap_swir": [2, 1, 0],
        "s2": [3, 2, 1],
    }

    split_percentages = [0.75, 0.1, 0.15]

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        product: str = "corine",
        split: str = "train",
        num_bands: int = 202,
        num_classes: int = 19,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        return_mask: bool = False,
        pin_memory: bool = False,
    ) -> None:
        assert sensor in self.valid_sensors, (
            f"Only supports one of {self.valid_sensors}, but found {sensor}."
        )
        self.sensor = sensor
        assert product in self.valid_products, (
            f"Only supports one of {self.valid_products}, but found {product}."
        )
        self.product = product
        assert split in self.valid_splits, (
            f"Only supports one of {self.valid_splits}, but found {split}."
        )
        self.split = split

        assert num_classes in [43, 19]

        self.root = root
        self.num_classes = num_classes
        self.transforms = transforms
        self.img_dir_name = self.image_root.format(self.sensor)
        self.mask_dir_name = self.mask_root.format(self.product)
        self.return_mask = return_mask
        self.num_bands = num_bands

        # Check if split file exists
        self.split_file = self.split_path.format(self.sensor, self.product, self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError("Split file does not exist. Please create it first.")

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

        if self.return_mask:
            sample = {
                "image": self._load_image(img_path),
                "label": self._load_raw_corine(mask_path),
            }

        else:
            sample = {
                "image": self._load_image(img_path),
                "label": self._load_mask(mask_path),
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_sample_with_mask(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        img_path, mask_path = self.sample_collection[index]

        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_raw_corine(mask_path),
        }

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
            image = torch.from_numpy(src.read())  # .float()
        return image

    def _load_raw_corine(self, path: str) -> Tensor:
        """Load the raw corine mask.

        Args:
            path: path to raw corine mask

        Returns:
            raw corine mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        return mask

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()

        # Apply corine_to_label mapping to the mask tensor
        mask = mask.apply_(lambda x: self.corine_to_label[x])

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            mask = mask.apply_(lambda x: self.label_converter[x])

        # Convert the mask to a mult-label encoding
        indices = np.unique(mask)
        if indices[-1] == 43:
            indices = indices[:-1]

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[indices] = 1.0

        return target

    def _onehot_labels_to_names(
        self, label_mask: "np.typing.NDArray[np.bool_]"
    ) -> list[str]:
        """Gets a list of class names given a label mask.

        Args:
            label_mask: a boolean mask corresponding to a set of labels or predictions

        Returns
            a list of class names corresponding to the input mask
        """
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.class_sets[self.num_classes][i])
        return labels

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

        image = sample["image"][
            self.rgb_indices[self.sensor]
        ].numpy()  # .permute(1, 2, 0)

        image = image.astype(int)

        image = image.transpose(1, 2, 0)

        label_vec = sample["label"].numpy().astype(np.bool_)
        label_names = self._onehot_labels_to_names(label_vec)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_vec = sample["prediction"].numpy().astype(np.bool_)
            prediction_names = self._onehot_labels_to_names(prediction_vec)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Labels: {', '.join(label_names)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(prediction_names)}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

    def plot_with_mask(
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

        image = sample["image"][
            self.rgb_indices[self.sensor]
        ].numpy()  # .permute(1, 2, 0)

        image = image.transpose(1, 2, 0)
        print(image.shape)

        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(mask, interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(self.cmap[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class EnMAPCorineBenchmarkDataModule(NonGeoDataModule):
    """Lightning DataModule for the EnMAP CORINE dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 224,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(EnMAPCorineBenchmark, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.aug = None

        mean = torch.tensor(np.load("data/statistics/mu.npy"))
        std = torch.tensor(np.load("data/statistics/sigma.npy"))

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

        mean = torch.tensor(np.load("data/statistics/mu.npy"))
        std = torch.tensor(np.load("data/statistics/sigma.npy"))

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


class EnMAPCorineDataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size)

        dataset_train = EnMAPCorineBenchmark(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = EnMAPCorineBenchmark(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = EnMAPCorineBenchmark(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
