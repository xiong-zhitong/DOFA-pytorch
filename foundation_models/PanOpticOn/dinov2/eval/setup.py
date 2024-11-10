# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch

from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights
from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from dinov2.utils.config import resolve_configs
import logging
import dinov2.eval.wrapper as wrapper
from functools import partial
import sys
import functools
import os
from omegaconf import OmegaConf, DictConfig

import dinov2.eval.models.DOFA.models_dwv as dofa

logger = logging.getLogger("dinov2")

MODEL_SELECTION = {
    'dofa_vitb_16': {
        'model_kwargs': {
            'arch': 'vit_base_patch16',
            'global_pool': False
        } ,
        'pretrained_weights': [
            {'path': '${oc.env:RESOURCE_DIR}/other_model_ckpts/DOFA/DOFA_ViT_base_e120.pth'}
        ]},
        'autocast_dtype_str': 'fp32',
}


@functools.lru_cache()
def setup_logger(name, filename=None, level=logging.DEBUG, to_sysout=False, simple_prefix=False, reset_logger=False):

    logger = logging.getLogger(name)
    if reset_logger:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(level)
    logger.propagate = False

    if simple_prefix:
        fmt_prefix = "%(asctime)s %(filename)s:%(lineno)s] "
        datefmt = "%H:%M:%S"
    else:
        fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
        datefmt = "%Y%m%d %H:%M:%S"
        
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if filename is None and not to_sysout:
        raise ValueError("Either filename or to_sysout must be set")

    if filename:
        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if to_sysout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def get_autocast_dtype(dtype_str):
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float

def parse_model_obj(model_obj, return_with_wrapper=True):
    if isinstance(model_obj, torch.nn.Module): # initialized model provided
        return model_obj

    elif isinstance(model_obj, DictConfig): # config provided
        config = model_obj

    elif isinstance(model_obj, dict) or isinstance(model_obj, OmegaConf):
        config = OmegaConf.create(model_obj)

    elif isinstance(model_obj, str):

        if model_obj.startswith('str:'): # model selection by string id
            config = MODEL_SELECTION[model_obj[4:]]
            config = OmegaConf.create(config)

        elif model_obj.endswith('.yaml'): # config file of model provided
            config = OmegaConf.load(model_obj)

        else: # train run_dir provided, pick model_final.pth and config.yaml
            run_dir = model_obj

            ckpt_path = os.path.join(run_dir, 'model_final.pth')
            ckpt = torch.load(ckpt_path, map_location='cpu') # just load to check keys

            if 'model' in ckpt: # full training ckpt
                pretrained_weights = dict(
                    path = os.path.join(run_dir, 'model_final.pth'),
                    checkpoint_key = 'model',
                    include = 'teacher',
                    exclude = 'cached_optical_embs',
                    prefix_map = {'teacher.backbone.': ''},)
                
            elif 'teacher' in ckpt: # only teacher ckpt, for sharing weights
                pretrained_weights = dict(
                    path = os.path.join(run_dir, 'model_final.pth'),
                    checkpoint_key = 'teacher',
                    include = 'backbone',
                    exclude = 'cached_optical_embs',
                    prefix_map = {'backbone.': ''},)
                
            else:
                raise ValueError(f'Unknown ckpt keys {ckpt.keys()}')

            model_kwargs = OmegaConf.load(os.path.join(run_dir, 'config.yaml')).student
            config = OmegaConf.create(dict(
                id = 'dinov2',
                pretrained_weights = [pretrained_weights],
                model_kwargs = model_kwargs))

    else:
        raise ValueError(f'Unknown model_obj type {type(model_obj)}')

    _model_wrapper = build_model_for_eval(config, return_with_wrapper=return_with_wrapper)
    return _model_wrapper

def parse_config_obj(config_obj):
    if isinstance(config_obj, list):
        configs = config_obj
    if isinstance(config_obj, str):
        if config_obj.endswith('.yaml'):
            configs = [config_obj]
        else: # directory of configs
            configs = [os.path.join(config_obj, f) 
                        for f in os.listdir(config_obj) if f.endswith('.yaml')]
    return configs

def build_model_for_eval(config, return_with_wrapper=True):
    id = config.id
    pretrained_weights = config.pretrained_weights
    model_kwargs = config.get('model_kwargs', {})
    autocast_dtype_str = config.get('autocast_dtype_str', 'fp32')

    # build model
    if id == 'dinov2':
        
        # add model defaults & adjust to fit with build_model_from_cfg
        cfgs = [
            OmegaConf.create(dinov2_default_config).student,
            model_kwargs] 
        build_cfg = {'student': OmegaConf.merge(*resolve_configs(cfgs))}
        build_cfg = OmegaConf.create(build_cfg)
        model, _ = build_model_from_cfg(build_cfg, only_teacher=True)
        _wrapper_class = wrapper.DINOv2Wrapper
        if 'autocast_dtype_str' in config:
            raise ValueError('autocast_dtype_str set to fp16 for DINOv2 model.')
        autocast_dtype_str = 'fp16'

    elif id == 'dofa':
        model = dofa.__dict__[model_kwargs.pop('arch')](**model_kwargs)
        _wrapper_class = wrapper.DOFAWrapper

    else: 
        raise ValueError(f"Unknown model id '{id}'")
    logger.info(f"Built model {id}")

    # load pretrained weights
    if len(pretrained_weights) > 0:
        load_pretrained_weights(model, pretrained_weights)
    else:
        logger.warning('No pretrained weights specified. Model will be randomly initialized.')
    autocast_dtype = get_autocast_dtype(autocast_dtype_str)

    if not return_with_wrapper:
        return model
    model.eval()
    model.cuda()
    return partial(_wrapper_class, feature_model=model, autocast_dtype=autocast_dtype)