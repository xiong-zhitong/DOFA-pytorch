"""Digital Typhoon Regression dataset."""


import torch
import kornia.augmentation as K
from torchgeo.datamodules import DigitalTyphoonDataModule


class RegDataAugmentation(torch.nn.Module):
    def __init__(self, split, size):
        super().__init__()

        # data already comes normalized between 0 and 1
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
        return x_out, batch["label"].float()

class DigitalTyphoonDataset:
    """Digital Typhoon dataset wrapper.
    
    # if automatic download/extraction fails, extract dataset in the root dir with `cat *.tar.gz.* | tar xvfz -`
    """
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

        # dataset has argument sequence length which actually dictates the number of channels
        # commonly in literature is used 3 channels and pretend it is RGB
        # sequence_length: length of the sequence to return
        dm = DigitalTyphoonDataModule(
            root=self.root_dir,
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