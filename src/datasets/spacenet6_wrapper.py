import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
import torch
from torchgeo.datasets import SpaceNet6
from torch.utils.data import random_split


class SegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, num_channels):
        """Initialize the data augmentation pipeline for the segmentation task.

        Args:
            split (str): The split of the dataset. Either 'train' or 'test'.
            size (int): The size of the image.
            num_channels (int): The desired number of input channels for the model.
        """
        super().__init__()

        self.num_channels = num_channels

        # TODO need to pick correct normalizaiton
        mean = torch.Tensor([0.5517])
        std = torch.Tensor([11.8478])

        if split == "train":
            self.transform = AugmentationSequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                data_keys=["image", "mask"],
            )
        else:
            self.transform = AugmentationSequential(
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
                data_keys=["image", "mask"],
            )

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        aug_sample = self.transform(sample)
        if self.num_channels != 1:
            aug_sample["image"] = aug_sample["image"].expand(
                -1, self.num_channels, -1, -1
            )

        # TODO find the correct wavelength depending on the sample path
        return aug_sample["image"].squeeze(0), aug_sample["mask"].squeeze(0).long()


class SpaceNet6Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.num_channels = config.num_channels

        self.val_split_pct = 0.1
        self.test_split_pct = 0.1

    def create_dataset(self):
        dataset = SpaceNet6(root=self.root_dir, split="train")
        generator = torch.Generator().manual_seed(0)
        # for now use a fixed random split
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )

        train_transform = SegDataAugmentation(
            split="train", size=self.img_size, num_channels=self.num_channels
        )
        eval_transform = SegDataAugmentation(
            split="test", size=self.img_size, num_channels=self.num_channels
        )

        train_dataset.transforms = train_transform
        val_dataset.transforms = eval_transform
        test_dataset.transforms = eval_transform

        return dataset_train, dataset_val, dataset_test
