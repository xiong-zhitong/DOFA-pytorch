from dinov2.utils.data import load_ds_cfg, extract_wavemus
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torch
from torch.functional import F
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import logging
import warnings
from tifffile import imread
import datetime
import traceback
from torchvision.io import read_image

logger = logging.getLogger("dinov2")

class SatlasDataset(Dataset):

    # S2 from USAT, L9 calculated internally (10% subset), S1 from MMEarth: average of ascending and descending orbits
    MEAN_VALUES:dict = {
        's2:b04': 0.3475, 's2:b03': 0.3547, 's2:b02': 0.3804, 's2:b05': 0.3038, 's2:b06': 0.3843, 's2:b07': 0.4232, 's2:b08': 0.4157, 's2:b11': 0.3687, 's2:b12': 0.2847,
        'landsat:b1': 0.4089, 'landsat:b2': 0.3725, 'landsat:b3': 0.3479, 'landsat:b4': 0.3442, 'landsat:b5': 0.6330, 'landsat:b6': 0.5025, 'landsat:b7': 0.3665, 'landsat:b8': 0.3427, 'landsat:b9': 0.0702, 'landsat:b10': 0.9373, 'landsat:b11': 0.9399,
        's1:vv': -11.716893758091746, 's1:vh': -19.117699039866956
    }
    STD_VALUES:dict = {
        's2:b04': 0.2394, 's2:b03': 0.1936, 's2:b02': 0.1836, 's2:b05': 0.1425, 's2:b06': 0.1434, 's2:b07': 0.1554, 's2:b08': 0.1526, 's2:b11': 0.1472, 's2:b12': 0.1264,
        'landsat:b1': 0.1701, 'landsat:b2': 0.1799, 'landsat:b3': 0.1923, 'landsat:b4': 0.2224, 'landsat:b5': 0.2728, 'landsat:b6': 0.2644, 'landsat:b7': 0.2348, 'landsat:b8': 0.2031, 'landsat:b9': 0.0415, 'landsat:b10': 0.1482, 'landsat:b11': 0.141,
        's1:vv': 5.0845347280556945, 's1:vh': 6.39358137263047
    }
    # TODO: Check if Satlas S1 is in dB or not!!!

    s2_bandname_map = {band: band for band in ['b05', 'b06', 'b07', 'b08', 'b11', 'b12']}
    s2_bandname_map.update({'b04': 'tci', 'b03': 'tci', 'b02': 'tci'})
    s1_bandname_map = {band: band for band in ['vv', 'vh']}
    landsat_bandname_map = {band: band for band in ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11']}

    
    #local to yaml filename mapping
    sensor_name_mapping: dict = {'s2' : 'satlas_s2',  'landsat': 'satlas_landsat', 's1': 'satlas_s1'}
    RGB_CHANNELS: dict = {
        'satlas_s2': [1, 2, 3],
        'satlas_landsat': [4, 2, 1], 
        'satlas_s1': [0, 1, 1],  
    }
    #names of bands in order, in the file paths
    


    def __init__(self, root='${RESOURCE_DIR}/datasets',
                 num_sens:int=2,
                 normalize:bool=True, 
                 keep_sensors=['s1', 's2', 'landsat'],
                 return_rgb=False,
                 transform=None,
                 max_tries_load_img=3,
                 full_spectra=False):
        """

        :param num_sens: Number of sensors to use. Default is 2.
        :param normalize: Normalize the image. Default is True.
        :param root: Root directory of the dataset. Default is '${RESOURCE_DIR}/datasets'.
        :param keep_sensors: List of sensors to keep. Default is ['s1', 's2', 'landsat'].
        :param return_rgb: Return RGB image. Default is False.
        :param transform: Transform to apply to the image. Default is None.
        :param max_tries_load_img: Maximum number of tries to load the image. Default is 3.
        :param full_spectra: Use full spectral embedding. Default is False.
        """

        self.MEAN = {
            "satlas_s2": {band: torch.tensor(self.MEAN_VALUES[f's2:{band}']) for band in self.s2_bandname_map.keys()},
            "satlas_landsat": {band: torch.tensor(self.MEAN_VALUES[f'landsat:{band}']) for band in self.landsat_bandname_map.keys()},
            "satlas_s1": {band: torch.tensor(self.MEAN_VALUES[f's1:{band}']) for band in self.s1_bandname_map.keys()}
        }

        self.STD = {
            "satlas_s2": {band: torch.tensor(self.STD_VALUES[f's2:{band}']) for band in self.s2_bandname_map.keys()},
            "satlas_landsat": {band: torch.tensor(self.STD_VALUES[f'landsat:{band}']) for band in self.landsat_bandname_map.keys()},
            "satlas_s1": {band: torch.tensor(self.STD_VALUES[f's1:{band}']) for band in self.s1_bandname_map.keys()}
        }

        root = os.path.expandvars(root)
        self.transform = transform
        self.return_rgb = return_rgb

        metadata_path = os.path.join(root, 'metadata_v2/fmow_iwm_onid_3sensors_all.parquet')
        self.df = pd.read_parquet(metadata_path)

        # self.faulty_imgs_file = faulty_imgs_file or '.'.join(metadata_path.split('.')[:-1]) + '_faulty_imgs'
        self.max_tries_load_img = max_tries_load_img

        # load dataset metainfo
        ds_names = self.sensor_name_mapping.values()
        self.chn_ids = {k: extract_wavemus(load_ds_cfg(k), full_spectra) for k in ds_names } 

        self.M = num_sens
        self.normalize = normalize
        self.root = root
        assert all(sensor in self.sensor_name_mapping.keys() for sensor in keep_sensors), f'Invalid sensor name in {keep_sensors}'
        self.keep_sensors = keep_sensors

        if self.normalize:
            logger.info('Building normalization transforms')
            self.channelwise_transforms = self._build_ch_transforms()
        

    def _build_ch_transforms(self):
        channelwise_transforms = {}
        for sensor in self.MEAN.keys():
            #create a new key for the sensor if it doesn't exist
            if sensor not in channelwise_transforms:
                channelwise_transforms[sensor] = {}
            for band in self.MEAN[sensor].keys():
                channelwise_transforms[sensor][band] = transforms.Normalize(self.MEAN[sensor][band], self.STD[sensor][band])
        return channelwise_transforms

    def __len__(self):
        return len(self.df)

    def log_stats(self):
        sensor_counts = {sensor: 0 for sensor in self.sensor_name_mapping.keys()}
        for sensor in self.sensor_name_mapping.keys():
            sensor_counts[sensor] = self.df['sensor'].apply(lambda x: sensor in x).sum()
        logger.info(f'Dataset size: {self.__len__()}, sensor counts: {sensor_counts}')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        def _get_idx():
            # select only the ids of the sensors that are in the keep_sensors list
            valid_sensor_idxs = [i for i, sensor in enumerate(row['sensor']) if sensor in self.keep_sensors]

            if len(valid_sensor_idxs) < self.M:
                raise ValueError('Not enough sensors')  
            
            idxs = np.random.choice(valid_sensor_idxs, self.M, replace=False)
            return idxs

        # load images with retries if failed
        tries = 0
        while tries < self.max_tries_load_img:
            idxs = _get_idx()
            imgs = []
            for i in idxs:
                try: 
                    imgs.append(self._load_img(row, i))
                except Exception as e:
                    raise ValueError(f'Error loading image: {e}')

            if len(imgs) == len(idxs):
                break
            tries += 1

        if tries == self.max_tries_load_img:
            raise ValueError(f'Failed to load images in {self.max_tries_load_img} tries')

        if self.normalize:
            for i in idxs:
                img_list = imgs.pop(0)
                sensor = self.sensor_name_mapping[row['sensor'][i]]
                
                for j, band in enumerate(self.MEAN[sensor].keys()):
                    if sensor == 'satlas_s1': #hack right now since s1 values are 0-1
                        img_list[j] -= 0.5
                    else:
                        img_list[j] = self.channelwise_transforms[sensor][band](img_list[j])
                        
                imgs.append(img_list)

        ds_names = [self.sensor_name_mapping[row['sensor'][i]] for i in idxs]
        chn_ids  = [self.chn_ids[ds] for ds in ds_names]
        chn_ids_list = []
        #chn_ids needs to be a list of len C each with a tensor of size (1,2)
        for tensor in chn_ids:
            tensor_list = [tensor[i].unsqueeze(0) for i in range(tensor.shape[0])]
            chn_ids_list.append(tensor_list)


        out = [dict(
            imgs = imgs[i],
            chn_ids = chn_ids_list[i],
        ) for i in range(self.M)]

        if self.transform:
            out = self.transform(out)
        return out
                    

    def _load_img(self, row, idx):
        sensor = row['sensor'][idx]
        time = row['time'][idx]

        img_list = []
        if sensor == 's1' :
            for band in self.s1_bandname_map.keys():
                path = os.path.join(self.root, f'sentinel1/{time}/{band}/{row["id"]}.png')
                img = read_image(path)
                img_list.append((img.type(torch.float) / 255.))
            # print(f's1: {len(img_list)}')

        elif sensor == 'landsat':
            for band in self.landsat_bandname_map.keys():
                path = os.path.join(self.root, f'landsat/{time}/{band}/{row["id"]}.png')
                img = read_image(path)
                img_list.append((img.type(torch.float) / 255.))
            # print(f'l9: {len(img_list)}')

        elif sensor == 's2':
            #find all the uinque values in self.s2_bandname_map.values() 
            for band in set(self.s2_bandname_map.values()):
                path = os.path.join(self.root, f'sentinel2/{row["s2_dir"]}/{band}/{row["id"]}.png')
                img = read_image(path)
                if img.shape[0] == 3:
                    for i in range(img.shape[0]):
                        img_list.append((img[i].type(torch.float) / 255.).unsqueeze(0)) #convert to [0-1] range because our normalizations params are in that range
                else:
                    img_list.append(img.type(torch.float) / 255.)#convert to [0-1] range because our normalizations params are in that range
                
            # print(f's2: {len(img_list)}')
        return img_list
        

