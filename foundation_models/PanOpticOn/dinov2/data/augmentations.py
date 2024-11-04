# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

""" 
Design: 
- augmentation either on a list of channels or on a standard image
"""

import logging

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from random import shuffle
import numpy as np
from torchvision.transforms.functional import pad, resized_crop
import torch
from copy import deepcopy
from torch import Tensor

from typing import List

logger = logging.getLogger("dinov2")


def make_augmentation(cfg):
    """ builds augmentations and also returns metainfo about crops """
    cfg = deepcopy(cfg)
    ret, ids = [], []

    for trf_cfg in cfg:
        id = trf_cfg.pop('id')
        ids.append(id)
        if id == 'DataAugmentationDINO':
            trf = DataAugmentationDINO(**trf_cfg)
    
        elif id == 'ChnSpatialAugmentation':
            trf = ChnSpatialAugmentation(**trf_cfg)

        elif id == 'ChnSpatialAugmentationV2':
            trf = ChnSpatialAugmentationV2(**trf_cfg)

        # elif id == 'ListCenterCrop':
        #     trf = ListCenterCrop(**trf_cfg)
        # elif id == 'Listify':
        #     trf = Listify(**trf_cfg)
        elif id == 'ChnSelect':
            trf = ChnSelect(**trf_cfg)
        elif id == 'Resize':
            trf = Resize(**trf_cfg)
        else:
            raise ValueError(f"Augmentation {id} not supported")
        ret.append(trf)
    logger.info(f"Augmentations in order: {ids}")

    ret = transforms.Compose(ret)
    return ret


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f'id: DataAugmentationDINO')
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class ChnSpatialAugmentation:
    def __init__(
        self,
        global_crops_number,
        global_crops_scale,
        local_crops_number,
        local_crops_scale,
        global_crops_size=[8,224],
        local_crops_size=[3,96],
        determinisitc_subset = False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.determinisitc_subset = determinisitc_subset
        self.global_crops_number = global_crops_number

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f'id: ChnSpatialAugmentation')
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"determinisitc_subset: {determinisitc_subset}")
        logger.info("###################################")

        # geometric augmentations
        self.geometric_augmentation_global = transforms.Compose([
            ListRandomChoice(global_crops_size[0], determinisitc_subset=determinisitc_subset),
            ListRandomResizeCrop(global_crops_size[1], global_crops_scale),
            ChnPad(global_crops_size[0]),
        ])
        self.geometric_augmentation_local = transforms.Compose([
            ListRandomChoice(local_crops_size[0], determinisitc_subset=determinisitc_subset),
            ListRandomResizeCrop(local_crops_size[1], local_crops_scale),
            ChnPad(local_crops_size[0]),
        ])

        # other augmentations
        pass # for now



    def __call__(self, x_dict: dict):
        # keys in x_dict: ds_names, imgs, gsds, time, wavelengths
        output = {}
        n_global_crops = len(x_dict['imgs']) # 1 global crop per sensor
        if self.global_crops_number - n_global_crops:
            i = 0
            for _ in range(self.global_crops_number - n_global_crops):
                for k in x_dict.keys():
                    x_dict[k].append(deepcopy(x_dict[k][i]))
                i = (i+1) % n_global_crops
            n_global_crops = self.global_crops_number

        n_local_crops_total = self.local_crops_number
        assert n_local_crops_total % n_global_crops == 0, \
            f"Number of total local crops {n_local_crops_total} must be divisible by number of global crops {n_global_crops}"
        n_local_crops_per_global = n_local_crops_total // n_global_crops

        # expand each img list to channel list
        for sens in range(n_global_crops):
            for _ in range(len(x_dict['imgs'][sens])):
                img = x_dict['imgs'][sens].pop(0)
                for chn in img:
                    x_dict['imgs'][sens].append(chn.unsqueeze(0)) # 1xhxw

        # geometric augmentations
        global_crops = []
        local_crops = []
        for sens in range(n_global_crops):
            sample_dict = dict(
                imgs = x_dict['imgs'][sens], chn_ids = x_dict['chn_ids'][sens])

            global_crop = self.geometric_augmentation_global(sample_dict)
            local_crop = [self.geometric_augmentation_local(sample_dict) 
                          for _ in range(n_local_crops_per_global)]

            global_crops.append(global_crop)
            local_crops += local_crop

        # make outputs
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


class ChnSpatialAugmentationV2:

    def __init__(
        self,
        global_crops_number,
        global_crops_scale,
        local_crops_number,
        local_crops_scale,
        global_crops_size=224,
        local_crops_size=98,
        global_crops_spectral_size=[6,13],
        local_crops_spectral_size=[3,6],
        global_modes_probs = [0.8, 0.1, 0.1],
        local_modes_probs = [0.2, 0.8],
        color_jitter_args = dict(p=0.3, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    ):
        assert global_crops_number == 2, "Only 2 global crops supported"
        assert sum(global_modes_probs) == 1, "Global modes probs must sum to 1"
        assert sum(local_modes_probs) == 1, "Local modes probs must sum to 1"

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.global_crops_number = global_crops_number
        self.global_modes_probs = global_modes_probs
        self.local_modes_probs = local_modes_probs

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f'id: ChnSpatialAugmentationV2')
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"global_crops_spectral_size: {global_crops_spectral_size}")
        logger.info(f"local_crops_spectral_size: {local_crops_spectral_size}")
        logger.info(f"global_modes_probs: {global_modes_probs}")
        logger.info(f"local_modes_probs: {local_modes_probs}")
        logger.info("###################################")

        # crop modes
        class GlobalCropModes:
            SENSOR_SINGLE = 0
            SENSOR_MULTIPLE = 1
            CHN_MIX = 2
        self.global_crop_modes = GlobalCropModes

        class LocalCropModes:
            SENSOR_SINGLE = 0
            SENSOR_MULTIPLE = 1
            CHN_MIX = 2
        self.local_crop_modes = LocalCropModes


        # geometric augmentations
        self.chn_cut_global = ListChnCut(global_crops_spectral_size[1])
        self.chn_choice_global = ListChnChoice(*global_crops_spectral_size)
        self.geometric_augmentation_global = transforms.Compose([
            ListRandomResizeCrop(global_crops_size, global_crops_scale),
            RandomHVFlip(p=0.5),
        ])

        self.chn_choice_local = ListChnChoice(*local_crops_spectral_size)
        self.geometric_augmentation_local = transforms.Compose([
            ListRandomResizeCrop(local_crops_size, local_crops_scale),
            RandomHVFlip(p=0.5),
        ])

        # other augmentations

        color_jittering = ColorJitterRS(**color_jitter_args)

        self.global_transfo = transforms.Compose([
            color_jittering,
            ChnPad(global_crops_spectral_size[1]),
            ])
        self.local_transfo = transforms.Compose([
            color_jittering,
            ChnPad(local_crops_spectral_size[1]),
            ])

    def _listoflists2list(self, list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    def _append_data_objs(self, data_objs):
        output = {}
        for k in data_objs[0].keys():
            output[k] = self._listoflists2list([d[k] for d in data_objs])
        return output

    def __call__(self, dicts_list: List):
        """ the basic data object is a dict. Each key contains a list of equal length with
                imgs: [L, tensor(c,h,w)]
                chn_ids: [L, tensor(c,...)]
                [any other keys]: [L, tensor(c,...)]
            The input is a list of data_objs, one for each sensor
        """

        # global crops

        global_crops = []
        modes = np.random.choice(
            range(len(self.global_modes_probs)), 
            size = self.global_crops_number, 
            p = self.global_modes_probs, 
            replace = True)
        single_sensors_not_selected = set(range(len(dicts_list))) # ensure that single sensor not selected twice

        for mode in modes:

            if mode == self.global_crop_modes.SENSOR_SINGLE:
                idx = np.random.choice(list(single_sensors_not_selected), 1, replace=False)[0]
                single_sensors_not_selected.remove(idx)
                data_obj = dicts_list[idx]
                data_obj = self.chn_cut_global(data_obj)

            elif mode == self.global_crop_modes.SENSOR_MULTIPLE:
                nsens = np.random.randint(2, len(dicts_list)+1)
                idx = np.random.choice(range(len(dicts_list)), nsens, replace=False) # at least 2 sensors
                data_obj = [dicts_list[i] for i in idx]
                data_obj = self._append_data_objs(data_obj)
                data_obj = self.chn_cut_global(data_obj)

            elif mode == self.global_crop_modes.CHN_MIX:
                data_obj = self._append_data_objs(dicts_list)
                data_obj = self.chn_choice_global(data_obj)

            data_obj = self.geometric_augmentation_global(data_obj)
            data_obj = self.global_transfo(data_obj)
            global_crops.append(data_obj)

        # local crops

        local_crops = []
        modes = np.random.choice(
            range(len(self.local_modes_probs)), 
            size = self.local_crops_number, 
            p = self.local_modes_probs, 
            replace = True)

        for mode in modes:

            if mode == self.local_crop_modes.SENSOR_SINGLE:
                idx = np.random.choice(range(len(dicts_list)), 1, replace=False)[0]
                data_obj = dicts_list[idx]

            elif mode == self.local_crop_modes.SENSOR_MULTIPLE:
                nsens = np.random.randint(2, len(dicts_list)+1)
                idx = np.random.choice(range(len(dicts_list)), nsens, replace=False) # at least 2 sensors
                data_obj = [dicts_list[i] for i in idx]
                data_obj = self._append_data_objs(data_obj)

            elif mode == self.local_crop_modes.CHN_MIX:
                data_obj = self._append_data_objs(dicts_list)

            data_obj = self.chn_choice_local(data_obj)
            data_obj = self.geometric_augmentation_local(data_obj)
            data_obj = self.local_transfo(data_obj)
            local_crops.append(data_obj)

        # return

        output = {}
        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output





class ChnSelector:
    """ abstract class to implement transforms that subselect channels of a data object """

    def _get_indices(self, data_obj):
        chn_indices = [(s,i) for s in range(len(data_obj['imgs'])) 
                    for i in range(len(data_obj['imgs'][s]))]
        return chn_indices

    def _apply_selection(self, data_obj, chn_indices):
        keys = data_obj.keys()
        data_obj_out = {k: [] for k in keys}
        for chn_group_in in range(len(data_obj['imgs'])):
            idx = [i for s,i in chn_indices if s == chn_group_in]
            if len(idx) > 0:
                for k in keys:
                    data_obj_out[k].append(data_obj[k][chn_group_in][idx])
        return data_obj_out

class ListChnCut(ChnSelector):
    def __init__(self, size):
        self.size = size

    def __call__(self, data_obj):
        chn_indices = self._get_indices(data_obj)
        if len(chn_indices) <= self.size:
            return data_obj

        shuffle(chn_indices)
        chn_indices = chn_indices[:self.size]
        return self._apply_selection(data_obj, chn_indices)

class ListChnChoice(ChnSelector):
    def __init__(self, low, high):
        assert low >= 1
        self.low = low
        self.high = high

    def __call__(self, data_obj):
        chn_indices = self._get_indices(data_obj)
        
        nchns = np.random.randint(self.low, self.high+1)
        if nchns == len(chn_indices):
            return data_obj

        shuffle(chn_indices)
        chn_indices = chn_indices[:nchns]
        return self._apply_selection(data_obj, chn_indices)


class RandomHVFlip:
    def __init__(self, p=0.5):
        self.trf = transforms.Compose([
            transforms.RandomVerticalFlip(p=p),
            transforms.RandomHorizontalFlip(p=p),
        ])

    def __call__(self, data_obj):
        data_obj_out = {k: v for k,v in data_obj.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.trf(data_obj['imgs'])
        return data_obj_out

class ColorJitterRS:
    def __init__(self, p=0.3, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        trf = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        trf = transforms.RandomApply([trf], p=p)
        self.trf = trf

    def __call__(self, data_obj):
        data_obj_out = {k: v for k,v in data_obj.items() if k != 'imgs'}
        data_obj_out['imgs'] = self.trf(data_obj['imgs'].unsqueeze(1)).squeeze(1)
        return data_obj_out

class ListRandomChoice:
    """ randomly selects up to fixed number of channels"""
    def __init__(self, n, determinisitc_subset: bool = False, seed=42):
        self.n = n
        self.determinisitc_subset = determinisitc_subset
        self.generator = torch.Generator().manual_seed(seed)
    
    def _torch_random_choice(self, n, k):
        return torch.multinomial(torch.ones(n) / n, k, replacement=False, generator=self.generator)

    def __call__(self, x_dict: dict):
        img_list = x_dict['imgs']
        chn_ids = x_dict['chn_ids']

        if self.n == -1: # no subsetting
            return x_dict
        
        if self.determinisitc_subset: # subset first self.n channels
            up = min(len(img_list), self.n)
            img_list = img_list[:up]
            chn_ids = chn_ids[:up]

        else: # randomly subsample self.n channels
            ninput = len(img_list)
            nsamples = min(ninput, self.n)
            idx = self._torch_random_choice(ninput, nsamples)
            img_list = [img_list[i] for i in idx]
            chn_ids = chn_ids[idx]

        return dict(imgs=img_list, chn_ids=chn_ids)

class ListRandomResizeCrop:
    """ takes list of different shape images and outputs RandomResizedCrop of same relative location """
    def __init__(self, 
                 size, 
                 scale, 
                 ratio = (3.0 / 4.0, 4.0 / 3.0),
                 antialias = True,
                 interpolation=transforms.InterpolationMode.BICUBIC,
                 ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.antialias = antialias
        self.interpolation = interpolation

    def __call__(self, x_dict: dict):
        img_list = x_dict['imgs']

        # get crop for largest in list
        idx_largest = np.argmax([img.shape[1] for img in img_list])
        i, j, h, w = transforms.RandomResizedCrop.get_params(img_list[idx_largest], self.scale, self.ratio)

        # crop all images (not in_place)
        img_largest_shape = img_list[idx_largest].shape
        ret = []
        for idx in range(len(img_list)):
            img = img_list[idx]
            if idx != idx_largest:
                
                # Calculate the crop parameters for the current image
                scale_x = img.shape[-1] / img_largest_shape[-1]
                scale_y = img.shape[-2] / img_largest_shape[-2]
                i_scaled, j_scaled = int(np.round(i * scale_y)), int(np.round(j * scale_x))
                h_scaled, w_scaled = int(np.ceil(h * scale_y)), int(np.ceil(w * scale_x))

                img_out = resized_crop(img, i_scaled, j_scaled, h_scaled, w_scaled,
                                    [self.size, self.size], self.interpolation, antialias=self.antialias)
      
            else:
                img_out = resized_crop(img, i, j, h, w, 
                                    [self.size, self.size], self.interpolation, antialias=self.antialias)
            ret.append(img_out)

        ret = torch.cat(ret, dim=0)
        chn_ids = x_dict['chn_ids']
        if isinstance(chn_ids, list):
            chn_ids = torch.cat(chn_ids, dim=0)
        return dict(imgs=ret, chn_ids=chn_ids)
    
# class ListCenterCrop:
#     """ upsample all images to largest image and take center crop of stacked images"""

#     def __init__(self, size, antialias: bool = True):
#         self.size = size
#         self.crop = transforms.CenterCrop(size)
#         self.antialias = antialias

#     def __call__(self, x_dict):
#         img_list = x_dict['imgs']
#         idx_largest = np.argmax([img.shape[1] for img in img_list])
#         shape_largest = img_list[idx_largest].shape
#         assert max(shape_largest[1:]) >= self.size, f'{shape_largest}, {self.size}'

#         ret = []
#         for idx in range(len(img_list)):
#             img = img_list[idx] # s.t. not in-place
#             if idx != idx_largest:
#                 img = torch.nn.functional.interpolate(
#                     img.unsqueeze(0).float(), 
#                     size=(shape_largest[-2], shape_largest[-1]), 
#                     mode='bilinear',
#                     antialias=self.antialias).squeeze(0)
#             ret.append(img)

#         img = self.crop(torch.cat(ret, dim=0))
#         return dict(imgs=img, chn_ids=x_dict['chn_ids'])
    
# class Listify:
#     def __call__(self, x_dict):
#         x_dict['imgs'] = [x_dict['imgs']]
#         return x_dict

class ChnPad:
    """ pads channels to same size """
    def __init__(self, size):
        self.size = size

    def __call__(self, x_dict):
        if self.size < 0:
            return x_dict
        img: Tensor = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        device = img.device

        c, h, w = img.shape
        spec_masks = torch.zeros(c, device=device)
        if c < self.size:
            img = torch.cat([img, torch.zeros(self.size - c, h, w, device=device)])
            # Check the number of dimensions
            if chn_ids.ndimension() == 1:
                chn_ids = torch.cat([chn_ids, torch.full((self.size - c,), chn_ids[-1])])
            elif chn_ids.ndimension() == 2:
                # Repeat the last row to fill the remaining size
                last_row = chn_ids[-1].unsqueeze(0).repeat(self.size - c, 1)
                chn_ids = torch.cat([chn_ids, last_row], dim=0)
                
            spec_masks = torch.cat([spec_masks, torch.ones(self.size - c, device=device)])
        spec_masks = spec_masks.bool()
        return dict(imgs=img, chn_ids=chn_ids, spec_masks=spec_masks)
    

class ChnSelect:
    def __init__(self, idxs):
        self.idxs = idxs

    def __call__(self, x_dict):
        imgs = x_dict['imgs'][self.idxs]
        chn_ids = x_dict['chn_ids'][self.idxs]
        return dict(imgs=imgs, chn_ids=chn_ids)
    
class Resize:
    def __init__(self, *args, **kwargs):
        kwargs['antialias'] = kwargs.get('antialias', True)
        self.resize = transforms.Resize(*args, **kwargs)

    def __call__(self, x_dict):
        img = x_dict['imgs']
        img = self.resize(img)
        return dict(imgs=img, chn_ids=x_dict['chn_ids'])