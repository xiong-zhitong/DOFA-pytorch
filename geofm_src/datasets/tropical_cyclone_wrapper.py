"""Tropical Cyclone Regression dataset."""


import torch
import kornia.augmentation as K
from torchgeo.datamodules import TropicalCycloneDataModule


class RegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        super().__init__()

        # image statistics for the dataset
        # input_mean = torch.Tensor([0.28154722, 0.28071895, 0.27990073])
        # input_std = torch.Tensor([0.23435517, 0.23392765, 0.23351675])

        # data already comes normalized between 0 and 1
        mean = torch.Tensor([0.0])
        std = torch.Tensor([1.0])

        target_mean = torch.Tensor([50.54925])
        target_std = torch.Tensor([26.836512])

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
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        # normalize the target
        target = (batch["label"].float() - target_mean) / target_std
        return x_out, target

class TropicalCycloneDataset:
    """Tropical Cyclone dataset wrapper."""
    def __init__(self, config) -> None:
        """Initialize the dataset wrapper.
        
        Args:
            config: Config object for the dataset, this is the dataset config
        """
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        """Create dataset splits for training, validation, and testing."""
        train_transform = RegDataAugmentation(split="train", size=self.img_size)
        eval_transform = RegDataAugmentation(split="test", size=self.img_size)

        dm = TropicalCycloneDataModule(
            root=self.root_dir, download=True
        )
        # use the splits implemented in torchgeo
        dm.setup('fit')
        dm.setup('test')

        dataset_train = dm.train_dataset
        dataset_val = dm.val_dataset
        dataset_test = dm.test_dataset

        dataset_train.dataset.transforms = train_transform
        dataset_val.dataset.transforms = eval_transform
        dataset_test.dataset.transforms = eval_transform

        return dataset_train, dataset_val, dataset_test