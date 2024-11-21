""" benchmark how long reloading the dataloader takes"""
from omegaconf import OmegaConf
from dinov2.data import collate_data_and_cast, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_dataset
import torch
from functools import partial
from timeit import timeit
from time import time

cfg = OmegaConf.load('/home/lewaldm/code/PanOpticOn/dinov2/configs/rs_ssl_default_config.yaml')
# cfg = OmegaConf.load('/home/lewaldm/code/PanOpticOn/dinov2/configs/increment.yaml')
# cfg = OmegaConf.merge(default_cfg, cfg)

dataset, crops_meta = make_dataset(cfg.train.dataset, with_crops_meta=True)

img_size = crops_meta["global_crops_size"]
patch_size = cfg.student.patch_size
n_tokens = (img_size // patch_size) ** 2
mask_generator = MaskingGenerator(
    input_size=(img_size // patch_size, img_size // patch_size),
    max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
)

inputs_dtype = torch.half
start_iter = 0

collate_fn = partial(
    collate_data_and_cast,
    mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
    mask_probability=cfg.ibot.mask_sample_probability,
    n_tokens=n_tokens,
    mask_generator=mask_generator,
    dtype=inputs_dtype,
    global_smask_absolute_tuple=cfg.train.global_smask_absolute_tuple,
    local_smask_absolute_tuple=cfg.train.local_smask_absolute_tuple,
)

sampler_type = SamplerType.INFINITE





data_loader = make_data_loader(
    dataset=dataset,
    batch_size=cfg.train.batch_size_per_gpu,
    num_workers=cfg.train.num_workers,
    shuffle=True,
    seed=start_iter,  # TODO: Fix this -- cfg.train.seed
    sampler_type=sampler_type,
    sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
    drop_last=cfg.train.drop_last,
    pin_memory=cfg.train.pin_memory,
    collate_fn=collate_fn,
    persistent_workers=True,
)



def fun(data_loader, n_samples):
    iter_time = time()
    iterable = iter(data_loader)
    iter_time = time() - iter_time

    data_time = time()
    for i in range(n_samples):
        batch = next(iterable)
    data_time = time() - data_time

    return iter_time, data_time

n_ben = 5
n_batches = 3
out = [fun(data_loader, 3) for _ in range(n_ben)]
iter_time = [o[0] for o in out]
data_time = [o[1] for o in out]
print(f"iter_time: {sum(iter_time)/n_ben}")
print(f"data_time: {sum(data_time)/n_batches/n_ben}")