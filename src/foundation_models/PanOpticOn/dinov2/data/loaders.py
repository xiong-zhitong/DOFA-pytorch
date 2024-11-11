# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from dotenv import load_dotenv


load_dotenv()
import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar

import torch
from torch.utils.data import Sampler, ConcatDataset

from dinov2.data.datasets.benv2 import BigEarthNetv2Wrapper
from dinov2.data.datasets.geobench import GeobenchDataset
from dinov2.data.augmentations import make_augmentation
from torch.utils.data import Subset

from .datasets import DummyDataset, FmowDataset
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from copy import deepcopy
from dinov2.data.datasets.mmearth import MMEarthWrapper
from dinov2.data.datasets.satlas import SatlasDataset

import random


logger = logging.getLogger("dinov2")

class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


class ConcatDatasetTrf(ConcatDataset):
    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform

    def __getitem__(self, idx):
        x = super().__getitem__(idx)
        if self.transform is not None:
            x = self.transform(x)
        return x

def make_dataset(cfg, dino_augm_cfg=None, seed=42, return_out_mods=False):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    cfg = deepcopy(cfg)
    id = cfg.pop('id')
    transform_cfg = cfg.pop('transform', [])
    subset = cfg.pop('subset', -1)

    logger.info(f'Building dataset "{id}" ...')

    # build transform
    if dino_augm_cfg is not None:
        transform_cfg.append(dino_augm_cfg)
    transform = make_augmentation(transform_cfg)

    # build datasets for train
    if id == 'DummyDataset':
        ds = DummyDataset(**cfg, transform=transform)
        ds_modalities_out = 1

    elif id == 'FmowDataset':
        ds = FmowDataset(**cfg, transform=transform)
        ds_modalities_out = ds.M

    elif id == 'MMEarth':
        ds = MMEarthWrapper(**cfg, transform=transform)
        ds_modalities_out = len(ds.modalities)
        if 'month' in ds.modalities:
            ds_modalities_out -= 1

    elif id == 'SatlasDataset':
        ds = SatlasDataset(**cfg, transform=transform)
        ds_modalities_out = ds.M

    elif id == 'ConcatDataset':
        datasets = cfg.pop('datasets')
        tuples = [make_dataset(d, return_out_mods=True) for d in datasets]
        datasets = [t[0] for t in tuples]
        out_mods = [t[1] for t in tuples]
        for i in range(1, len(out_mods)):
            assert out_mods[i] == out_mods[0], f"all out_mods must be the same, got {out_mods}"
        ds = ConcatDatasetTrf(datasets, transform=transform)
        ds_modalities_out = out_mods[0]

    # build datasets for eval
    elif 'geobench' in id:
        ds = GeobenchDataset(ds_name=id.split('.')[1], **cfg, transform=transform)

    elif id == 'benv2':
        ds = BigEarthNetv2Wrapper(**cfg, transform=transform)

    else:
        raise ValueError(f'Unsupported dataset "{id}"')
    logger.info(f'Built dataset "{id}" with #samples {len(ds)}')

    # subset
    if subset > 0:
        def sample_indices(n, k):
            generator = torch.Generator().manual_seed(seed)
            return torch.multinomial(torch.ones(n) / n, k, replacement=False, generator=generator).tolist()

        if isinstance(subset, float):
            assert 0.0 < subset <= 1.0, 'Float subset must be in range (0, 1].'
            if subset < 1.0:
                subset_indices = sample_indices(len(ds), int(len(ds)*subset))
                ds = Subset(ds, subset_indices)
        elif isinstance(subset, int):
            assert subset > 0, 'Int subset must be greater than 0.'
            if subset < len(ds):
                subset_indices = sample_indices(len(ds), subset)
                ds = Subset(ds, subset_indices)
            else:
                sampler = EpochSampler(size=subset, sample_count=len(ds), shuffle=False, seed=seed)
                subset_indices = sampler._get_iterable().tolist()
                ds = Subset(ds, subset_indices)
        else:
            raise ValueError(f'Unsupported subset type "{type(subset)}"')
        logger.info(f'Got subset={subset}, subsampled dataset to #samples {len(ds)} ')
    
    # check modalities
    if dino_augm_cfg is not None:
        if ds_modalities_out != dino_augm_cfg.global_crops_number:
            logger.warning(f"Dataset modalities ({ds_modalities_out}) != DINO augm modalities ({dino_augm_cfg.global_crops_number})")

    if return_out_mods:
        return ds, ds_modalities_out
    return ds


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
    drop_last: bool = False
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
        drop_last=drop_last
    )

    logger.info(f"using PyTorch data loader (bsz={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}, persistent_workers={persistent_workers})")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
