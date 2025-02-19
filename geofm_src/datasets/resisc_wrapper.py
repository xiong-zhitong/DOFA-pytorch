import kornia.augmentation as K
import torch
from torchgeo.datasets import RESISC45


class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
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
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"]


class Resics45Dataset:
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size)

        dataset_train = RESISC45(
            root=self.root_dir, split="train", transforms=train_transform
        )
        dataset_val = RESISC45(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = RESISC45(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
