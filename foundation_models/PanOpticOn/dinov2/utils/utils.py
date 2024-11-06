# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
import re

logger = logging.getLogger("dinov2")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key='model', ):
    full_state_dict = {}
    for cfg in pretrained_weights:
        ckpt_path = cfg['path']
        logger.info(f"Processing pretrained weights from {ckpt_path}")
        if urlparse(ckpt_path).scheme:  # If it looks like an URL
            state_dict = torch.hub.load_state_dict_from_url(ckpt_path, map_location="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        if 'checkpoint_key' in cfg:
            checkpoint_key = cfg['checkpoint_key']
        if checkpoint_key is not None and checkpoint_key in state_dict:
            logger.info(f"Take key '{checkpoint_key}' in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        if 'include' in cfg:
            patterns = cfg.include.split(',')
            keys_to_include = []
            for pattern in patterns:
                keys_to_include += [k for k in state_dict.keys() if re.search(pattern, k)]
            state_dict = {k: v for k, v in state_dict.items() if k in keys_to_include}

        if 'exclude' in cfg:
            patterns = cfg.exclude.split(',')
            keys_to_drop = []
            for pattern in patterns:
                keys_to_drop += [k for k in state_dict.keys() if re.search(pattern, k)]
            state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_drop}
        
        if 'prefix_map' in cfg:
            for k,v in cfg.prefix_map.items():
                state_dict = {v + kk.removeprefix(k): vv for kk, vv in state_dict.items() if kk.startswith(k)}
            logger.info(f'Applied prefix load map: {cfg.prefix_map}')

        logger.info(f'From "{ckpt_path}" selected keys are: {list(state_dict.keys())}')
        full_state_dict.update(state_dict)

    msg = model.load_state_dict(full_state_dict, strict=False)

    # pretty print of message
    pstr = ''
    for k in ['missing_keys', 'unexpected_keys']:
        pstr += f'{k} (len={len(msg.__getattribute__(k))}):\n'
        for kk in msg.__getattribute__(k):
            pstr += f'  {kk}\n'
    logger.info(f"Loaded state_dict with msg:\n{pstr}")
    


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters, f'{len(self.schedule)}, {self.total_iters}'

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
