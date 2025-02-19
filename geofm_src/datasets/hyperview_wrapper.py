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
from typing import Optional, Callable
from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg, MaskTensor
from torchvision import transforms

# def resize_hyperspectral_tensor(img_tensor, output_shape=(150, 128, 128)):
#     img_tensor = img_tensor.unsqueeze(0)

#     resized_tensor = F.interpolate(
#         img_tensor, size=output_shape[1:], mode="bicubic", align_corners=True
#     )

#     resized_tensor = resized_tensor.squeeze(0)

#     return resized_tensor





class HyperviewBenchmark(NonGeoDataset):

    HYPERVIEW_MEAN = [ 290.7843,  290.3723,  286.6646,  292.1786,  299.0341,  312.8545,
         320.6057,  321.0374,  322.5741,  324.4897,  328.3440,  334.4489,
         339.1177,  351.7271,  363.4810,  369.6554,  377.4395,  385.3248,
         393.0646,  393.4557,  399.7940,  411.9576,  430.1181,  450.2287,
         461.3303,  468.1501,  477.8180,  483.6719,  492.9649,  499.4987,
         508.5567,  515.3029,  520.2290,  522.4949,  522.2994,  522.7948,
         528.0905,  531.5267,  537.5836,  544.4802,  551.8073,  555.8667,
         559.2421,  561.5034,  566.9113,  572.5939,  575.4571,  574.9772,
         577.4212,  580.6649,  585.6681,  589.2350,  591.1851,  591.1440,
         594.5833,  598.5197,  602.8950,  614.3005,  619.6977,  617.0872,
         611.0182,  603.3171,  598.6248,  597.0683,  600.6963,  595.6002,
         592.6815,  596.1755,  604.9003,  630.3424,  664.4611,  703.5816,
         742.1624,  775.1469,  801.4249,  825.0682,  856.0341,  884.6969,
         913.3508,  942.0082,  970.6569,  999.3099, 1027.4458, 1056.8407,
        1085.4407, 1114.2037, 1142.8521, 1169.8014, 1193.2913, 1212.1575,
        1226.2242, 1236.3239, 1244.7190, 1250.3289, 1254.9497, 1264.5237,
        1275.5308, 1287.0815, 1298.5231, 1308.8108, 1317.0856, 1322.9956,
        1326.9932, 1330.5647, 1336.4053, 1341.7981, 1345.5804, 1350.1261,
        1354.8489, 1359.4071, 1363.9717, 1368.5348, 1373.0947, 1377.6532,
        1381.8234, 1387.7123, 1390.3658, 1395.1348, 1399.6979, 1407.9180,
        1413.0568, 1404.8185, 1398.8970, 1402.4882, 1405.4640, 1407.9833,
        1407.9244, 1408.6720, 1403.3954, 1400.0491, 1394.9823, 1400.9719,
        1416.4968, 1423.0186, 1424.8911, 1429.3918, 1431.3179, 1433.4911,
        1434.8601, 1435.8971, 1435.0978, 1436.8961, 1437.9344, 1439.0907,
        1440.3364, 1441.6919, 1443.0356, 1443.9950, 1444.5940, 1445.2040]
    
    HYPERVIEW_STD = [ 184.6625,  184.4921,  182.4789,  185.6263,  189.6261,  197.5082,
         202.0506,  202.4590,  203.6059,  205.0439,  207.6424,  211.5326,
         214.5449,  222.0868,  229.2302,  233.0880,  237.8239,  242.5836,
         247.2846,  247.8141,  251.8614,  259.4138,  270.4821,  282.6596,
         289.5659,  293.8338,  299.7513,  303.4443,  309.1039,  313.0972,
         318.5313,  322.6151,  325.6525,  327.1701,  327.2935,  327.8849,
         331.3679,  333.7404,  337.6210,  341.9542,  346.5125,  349.0899,
         351.2251,  352.6437,  355.9573,  359.4665,  361.3185,  361.1541,
         362.8065,  364.9677,  368.1142,  370.2596,  371.3023,  371.0717,
         372.9365,  375.1887,  377.9156,  385.1665,  388.9590,  387.9171,
         384.7903,  380.7711,  378.7262,  378.7864,  382.1685,  379.7439,
         379.1682,  382.6469,  388.4000,  402.2303,  419.0303,  437.7646,
         457.3473,  476.8951,  496.5936,  518.0956,  545.7634,  574.8895,
         606.0114,  638.8474,  673.1370,  708.6809,  744.6009,  782.8761,
         821.8232,  861.3900,  900.5791,  936.6815,  966.9325,  989.7015,
        1005.0406, 1014.6472, 1021.6771, 1026.5959, 1030.5853, 1038.7617,
        1048.1011, 1057.2289, 1065.7727, 1072.9796, 1078.3110, 1081.6951,
        1083.6781, 1085.4855, 1089.0480, 1092.2404, 1094.1437, 1096.6281,
        1099.2626, 1101.7919, 1104.3225, 1106.8527, 1109.3929, 1111.9341,
        1114.2084, 1117.8269, 1118.8195, 1121.4557, 1123.9249, 1129.6223,
        1133.3379, 1126.6057, 1121.7826, 1124.5750, 1126.8206, 1128.5920,
        1128.2804, 1128.6567, 1124.3521, 1121.5509, 1117.1182, 1121.2203,
        1133.2728, 1138.4189, 1139.9752, 1143.5162, 1144.5846, 1145.2258,
        1144.5868, 1143.1881, 1140.1182, 1139.0898, 1137.4788, 1135.8882,
        1134.3488, 1132.7020, 1130.7249, 1128.3368, 1125.5746, 1122.6283]
    

    # Label stats
    TARGET_MEAN = torch.tensor([ 62.3122, 201.5815, 141.3911,   5.9815])
    TARGET_STD = torch.tensor([36.6041, 94.0852, 64.3309,  2.2037])
    # Max per label: tensor([325.0000, 625.0000, 400.0000,   7.8000], dtype=torch.float64)
    # Min per label: tensor([ 20.3000, 109.0000,  26.8000,   5.6000], dtype=torch.float64)

    valid_splits = ["train", "val", "test"]

    split_path = "splits/{}.csv"
    gt_file_path = "gt_{}.csv"

    rgb_indices = [43, 28, 10]
    split_percentages = [0.75, 0.1, 0.15] #train, val, test

    num_channels = 150

    keys = ["P", "K", "Mg", "pH"]

    def __init__(
        self,
        root: str,
        data_dir: str='train_data',
        split: str = "train",
        create_splits: bool = False,
        transforms: Optional[Callable] = None,
        normalize: bool = True,
        seed: int = 13, #optional, for splitting
    ) -> None:
        assert split in self.valid_splits, (
            f"Only supports one of {self.valid_splits}, but found {split}."
        )
        self.split = split
        self.normalize = normalize
        self.root = root
        self.img_path = os.path.join(self.root, data_dir)
        self.splits_path = os.path.join(self.root, self.gt_file_path.format(self.split))
        self.transforms = transforms
        #check if the img_path exists
        if create_splits:
            self.seed = seed
            self.split_train_val_test()
        else:
            # Check if split file exists, if not create it
            self.split_file = os.path.join(self.root, self.split_path.format(self.split))
            print(f'[HyperviewBenchmark] Building dataset for split: {self.split}')
            if os.path.exists(self.split_file):
                self.df = pd.read_csv(self.split_file)
            else:
                # self.split_train_val_test()
                raise ValueError("Split file does not exist. Please create it first.")
            
        if self.normalize:
            self.channelwise_transforms = self._build_ch_transforms()

    def split_train_val_test(self) -> list:
        """Split Train/Val/Test at the tile level."""

        from sklearn.model_selection import train_test_split
        from glob import glob

        np.random.seed(self.seed)

        if self.gt_path is not None:
            df = pd.read_csv(os.path.join(self.root, 'train_gt.csv'))

        file_paths = sorted(
            glob(os.path.join(self.img_path, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", ""))
        )
        df['file_paths'] = file_paths

        train_val_split = self.split_percentages[0] + self.split_percentages[1]
        train_val_df, test_df = train_test_split(df, test_size=self.split_percentages[2], random_state=self.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=self.split_percentages[1] / train_val_split, random_state=self.seed)

        #save the splits
        #make a new directory under the root directory called splits
        splits_dir = os.path.join(self.root, 'splits')
        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)


    def _build_ch_transforms(self):
        self.MEAN = torch.tensor(self.HYPERVIEW_MEAN)
        self.STD = torch.tensor(self.HYPERVIEW_STD)
        
        return transforms.Compose([
            transforms.Normalize(self.MEAN, self.STD),
        ])

    def read_image(self, img_path: str, return_mask: bool = True) -> np.ndarray:
        """Read image from .npz file."""
        with np.load(img_path) as npz:
            masked_arr = np.ma.MaskedArray(**npz)

        data = masked_arr.data
        if return_mask:
            mask = masked_arr.mask
            return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
        else:
            return torch.tensor(data, dtype=torch.float32)

        
    def load_gt(self, index: int):
        """Load labels for train set from the ground truth file.
        Args:
            file_path (str): Path to the ground truth .csv file.
        Returns:
            [type]: 2D numpy array with soil properties levels
        """
        row = self.df.iloc[index]
        targets = torch.tensor(row[self.keys].values.tolist())
        targets = (targets - self.TARGET_MEAN) / self.TARGET_STD
        return targets

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        file_path = self.df.iloc[index]["file_paths"]
        image, mask = self.read_image(file_path)
        label = self.load_gt(index)

        if self.normalize:
            image = self.channelwise_transforms(image)

        if self.transforms is not None:
            image = self.transforms((image, mask)) # the first transform is the mask tensor

        return image, label

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df)


    def plot(
        self,
        image: torch.Tensor,
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

        #use the rgb_indices to plot the rgb image
        rgb_image = image[self.rgb_indices].numpy()
        rgb_image = rgb_image.transpose(1, 2, 0)
        plt.imshow(rgb_image)
        plt.show()



    
class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, size, source_chn_ids, split="val", 
                 mean=None, std=None, band_ids=None, 
                 target_chn_ids=None, mask_image=True):
        super().__init__()

        flipH = K.RandomHorizontalFlip(p=0.5, keepdim=True)
        flipV = K.RandomVerticalFlip(p=0.5, keepdim=True)
        crop = K.RandomResizedCrop(_to_tuple(size), scale=(0.8, 1.0), keepdim=True) #, resample='bicubic')
        mask_tensor = MaskTensor(mask=mask_image)
        self.output_chn_ids = None
        
        # setup HS specific augmentations
        if band_ids is not None:
            chn_sample = ChannelSampler(band_ids)
            if source_chn_ids is not None:
                self.output_chn_ids = source_chn_ids[band_ids]
            else:
                raise ValueError("[ClsDataAugmentation] source_chn_ids must be provided if band_ids are provided")
        elif target_chn_ids is not None:
            chn_sim = ChannelSimulator(source_chn_ids=source_chn_ids, target_chn_ids=target_chn_ids)
            self.output_chn_ids = target_chn_ids

        else:
            self.output_chn_ids = source_chn_ids

        self.transforms = []
        if split == "train":
            if band_ids is not None:
                print(f'[ClsDataAugmentation: train] Sampling channels: {band_ids}')
                self.transforms.append(chn_sample)
            elif target_chn_ids is not None:
                print(f'[ClsDataAugmentation: train] Simulating channels: {target_chn_ids}')
                self.transforms.append(chn_sim)
            else:
                pass
            self.transforms.extend([mask_tensor, crop, flipH, flipV])
        else:
            if band_ids is not None:
                print(f'[ClsDataAugmentation: val/test] Sampling channels: {band_ids}')
                self.transforms.append(chn_sample)
            elif target_chn_ids is not None:
                print(f'[ClsDataAugmentation: val/test] Simulating channels: {target_chn_ids}')
                self.transforms.append(chn_sim)
            else:
                pass

            self.transforms.extend([mask_tensor, crop])

        self.transform = torch.nn.Sequential(*self.transforms)

    def get_chn_ids(self):
        return self.output_chn_ids

    @torch.no_grad()
    def forward(self, x):
        return self.transform(x)


class HyperviewDataset:
    def __init__(self, config):
        self.config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.band_ids = config.get("band_ids", None)
        self.ds_name = config.dataset_name
        self.target_ds_name = config.get('target_dataset_name', None)
        self.full_spectra = config.get("full_spectra", False) # only for panopticon
        self.source_chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.ds_name), True), dtype=torch.long) #load all bands
        if self.target_ds_name is not None:
            self.target_chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.target_ds_name), True), dtype=torch.long) #load all bands
        else:
            self.target_chn_ids = None
        self.mask_image = config.get("mask_image", True)

        #both band_ids and target_ds_name are mutually exclusive
        # assert (self.band_ids is not None and self.target_ds_name is None) or (self.band_ids is None and self.target_ds_name is not None), "Both band_ids and target_ds_name cannot be provided"

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size, band_ids=self.band_ids, source_chn_ids=self.source_chn_ids, target_chn_ids=self.target_chn_ids, mask_image=self.mask_image)
        eval_transform  = ClsDataAugmentation(split="test", size=self.img_size, band_ids=self.band_ids, source_chn_ids=self.source_chn_ids, target_chn_ids=self.target_chn_ids, mask_image=self.mask_image)


        # Override the config with the transformed channel ids
        output_chn_ids = train_transform.get_chn_ids() #provides the updated channel ids after augmentation
        if output_chn_ids is not None:
            self.config['wavelengths_mean_nm'] = output_chn_ids[:,0].tolist()
            if self.full_spectra:
                self.config['wavelengths_sigma_nm'] = output_chn_ids[:,1].tolist()

        dataset_train = HyperviewBenchmark(
            root=self.root_dir, split="train", transforms=train_transform,
        )
        dataset_val = HyperviewBenchmark(
            root=self.root_dir, split="val",  transforms=eval_transform,
        )
        dataset_test = HyperviewBenchmark(
            root=self.root_dir, split="test",  transforms=eval_transform,
        )

        return dataset_train, dataset_val, dataset_test
