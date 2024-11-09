"""BigEarthNetv2 dataset."""

import glob
import json
import os
from typing import Callable, Optional
import kornia.augmentation as K
from torch.utils.data import Subset, random_split
from torch import Generator

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from torch import Tensor
import pandas as pd
from torchvision import transforms
from torchgeo.datasets import BigEarthNet
from torchgeo.datasets.utils import download_url, extract_archive, sort_sentinel2_bands


class BigEarthNetv2(BigEarthNet):
    """BigEarthNetv2 dataset.
    
    Automatic download not implemented, get data from below link.
    """

    splits_metadata = {
        "train": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "val": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "test": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
    }
    metadata_locs = {
        "s1": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst", 
            "md5": "", #unknown
            "filename": "BigEarthNet-S1.tar.zst",
            "directory": "BigEarthNet-S1",
        },
        "s2": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst", 
            "md5": "", #unknown
            "filename": "BigEarthNet-S2.tar.zst",
            "directory": "BigEarthNet-S2",
        },
        "maps": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "", #unknown
            "filename": "Reference_Maps.zst",
            "directory": "Reference_Maps",
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(root=root, split=split, bands=bands, num_classes=num_classes, transforms=transforms, download=download, checksum=checksum)
    
        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])} 
        # self._verify()
        self.folders = self._load_folders()

    def get_class2idx(self, label:str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        pass


    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))

        def construct_folder_path(root, dir, patch_id, remove_last:int=2):
            tile_id = '_'.join(patch_id.split('_')[:-remove_last])
            return os.path.join(root, dir, tile_id, patch_id)

        folders = [
            {
                "s1": construct_folder_path(self.root, dir_s1, row['s1_name'] ,3),
                "s2": construct_folder_path(self.root, dir_s2, row['patch_id'],2),
                "maps": construct_folder_path(self.root, dir_maps, row['patch_id'],2),
            }
            for _, row in self.metadata.iterrows()
        ] 

        return folders

    def _load_map_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder_maps = self.folders[index]["maps"]
        paths_maps = glob.glob(os.path.join(folder_maps, "*_reference_map.tif"))
        paths_maps = sorted(paths_maps)
        return paths_maps


    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_map_paths(index)
        map = None
        for path in paths:
            with rasterio.open(path) as dataset:
                map = dataset.read(
                    # indexes=1,
                    # out_shape=self.image_size,
                    out_dtype="int32",
                    # resampling=Resampling.bilinear,
                )
        return torch.from_numpy(map).float()


    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """

        image_labels = self.metadata.iloc[index]['labels']

        # labels -> indices
        indices = [self.get_class2idx(label, level=self.num_classes) for label in image_labels]


        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target


class ClsDataAugmentation(torch.nn.Module):
    mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    maxs_raw = torch.tensor(
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    maxs = torch.tensor(
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )

    def __init__(self, split, size, bands="all"):
        super().__init__()

        if bands == 'all':
            mins = self.mins
            maxs = self.maxs
        elif bands == 's1':
            mins = self.mins[:2]
            maxs = self.maxs[:2]
        elif bands == 's2':
            mins = self.mins[2:]
            maxs = self.maxs[2:]
        elif bands == 'rgb':
            mins = self.mins[2:5].flip(dims=(0,)) # to get RGB order
            maxs = self.maxs[2:5].flip(dims=(0,))

        self.bands = bands
        self.mean = mins
        self.std = maxs - mins

        if split == 'train':
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=self.mean, std=self.std),
                K.Resize(size=size, align_corners=True),
            )

    @torch.no_grad()
    def forward(self, sample: dict[str, ]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple."""
        if self.bands == "rgb":
            sample["image"] = sample["image"][1:4, ...].flip(dims=(0,))
            # get in rgb order and then normalization can be applied
        x_out = self.transform(sample["image"]).squeeze(0)
        return x_out, sample["label"]


class BenV2Dataset():

    def __init__(self, config):

        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.bands = config.bands
        self.num_classes = config.num_classes

        if self.bands == 'rgb':
            # start with rgb and extract later
            self.input_bands = 's2'
        else:
            self.input_bands = self.bands


    def create_dataset(self):

        train_transform = ClsDataAugmentation(split="train", size=self.img_size, bands=self.bands)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size, bands=self.bands)

        dataset_train = BigEarthNetv2(root=self.root_dir, num_classes=self.num_classes, split="train", bands=self.input_bands, transforms=train_transform)


        num_subset_samples = int(0.1 * len(dataset_train))
        # Split the dataset into the subset and the remaining part
        subset_train, _ = random_split(dataset_train, [num_subset_samples, len(dataset_train) - num_subset_samples], generator=Generator().manual_seed(42))

        dataset_val = BigEarthNetv2(root=self.root_dir, num_classes=self.num_classes, split="val", bands=self.input_bands, transforms=eval_transform)
        dataset_test = BigEarthNetv2(root=self.root_dir, num_classes=self.num_classes, split="test", bands=self.input_bands, transforms=eval_transform)

        return subset_train, dataset_val, dataset_test
