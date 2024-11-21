# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

from omegaconf import OmegaConf

import dinov2.distributed as distributed
from dinov2.logging import setup_logging
from dinov2.utils import utils
from dinov2.configs import dinov2_default_config, dinov2_debug_config
from dotenv import load_dotenv
import random
import re

import wandb
from pathlib import Path

logger = logging.getLogger("dinov2")

def write_config(cfg, output_dir, name="config.yaml"):
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path

def resolve_configs(cfgs):
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver(
        name='list',
        resolver=lambda *args: list(args))
    OmegaConf.register_new_resolver(
        name='dict',
        resolver=lambda *args: {k: v for k,v in [i.split(':') for i in args]})
    OmegaConf.register_new_resolver(
        name='env',
        resolver=lambda arg: os.environ[arg])
    def _resolve(cfg: OmegaConf):
        if not isinstance(cfg, OmegaConf):
            cfg = OmegaConf.create(cfg)
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    return [_resolve(c) for c in cfgs]

def default_setup(args):
    distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("dinov2")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def apply_scaling_rules_to_cfg(cfg):  # to fix
    rules = cfg.optim.scaling_rule.split(',')
    for r in rules:
        if r == "sqrt_wrt_1024":
            base_lr = cfg.optim.base_lr
            cfg.optim.lr = base_lr
            cfg.optim.lr *= math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_global_size() / 1024.0)
            logger.info(f"sqrt scaling learning rate; base: {base_lr}, new: {cfg.optim.lr}")
        elif r == 'min_lr=lr':
            cfg.optim.min_lr = cfg.optim.lr
            logger.info(f'Set min_lr = lr = {cfg.optim.lr}')
        elif r == 'find_lr': # fix lr to lr, fix all other schedulers to their initial values
            cfg.optim.min_lr = cfg.optim.lr
            cfg.teacher.final_momentum_teacher = cfg.teacher.momentum_teacher
            cfg.teacher.warmup_teacher_temp_epochs = 0
            cfg.teacher.teacher_temp = cfg.teacher.warmup_teacher_temp
            cfg.optim.weight_decay_end = cfg.optim.weight_decay
            logger.info('Find lr mode. Set lr to lr, fix all other schedulers to their initial values')
        else:
            raise NotImplementedError(r)
    return cfg

def get_train_cfg_from_args(args):

    # merge configs
    cfgs = [
        OmegaConf.create(dinov2_default_config),
        OmegaConf.load(args.config_file),
        OmegaConf.load(args.env) if args.env else OmegaConf.create({}),
        OmegaConf.from_cli(args.opts),
        OmegaConf.create(dinov2_debug_config) if args.fastdevrun else OmegaConf.create({})]

    cfgs = resolve_configs(cfgs)
    cfg = OmegaConf.merge(*cfgs)

    # set output dir 
    output_dir = os.path.abspath(args.output_dir)
    if args.fastdevrun:
        output_dir = os.path.join(output_dir, str(random.getrandbits(128)))
    elif cfg.add_args:
        kv = [(a.split('=')[0], '='.join(a.split('=')[1:])) for a in args.opts]

        # drop non-hparam args
        to_drop = [
            'train.output_dir',
            'train.use_wandb',
            'add_args',
            'optim.epochs',
            'optim.scaling_rule',
            'eval.'] 
        opts = {k: v for k,v in kv if not any([re.search(d,k) for d in to_drop])} 

        # rename args
        name_map = {
            'train.batch_size_per_gpu': 'bsz',
            'optim.base_lr': 'lr',
            'optim.warmup_epochs': 'warmup',
            'optim.freeze_weights': 'frz',
            'optim.lr_multiplier': 'lrmul',
            'train.dino_augm.global_crops_size': 'gcsz-spat',
            'train.dino_augm.local_crops_size': 'lcsz-spat',
            'dino.loss_weight': 'wdino',
            'ibot.loss_weight': 'wibot',
            'dino.koleo_loss_weight': 'wkoleo',
            'train.OFFICIAL_EPOCH_LENGTH': 'OFFEPOCH',
            'ibot.separate_head': 'sephead',
            'train.dino_augm.global_modes_probs': 'gprobs',
            'train.dino_augm.local_modes_probs': 'lprobs',
            'train.dino_augm.color_jitter_args.p': 'cjitter-p',
            'train.dino_augm.color_jitter_args.brightness': 'cjitter-brgth',
            'train.dino_augm.color_jitter_args.contrast': 'cjitter-contr',
            'train.dino_augm.color_jitter_args.saturation': 'cjitter-sat',
            'train.dino_augm.color_jitter_args.hue': 'cjitter-hue',
            'train.dino_augm.global_crops_spectral_size': 'gcsz',
            'train.dino_augm.local_crops_spectral_size': 'lcsz',
            }
        opts = {name_map.get(k,k): v for k,v in opts.items()} 

        opts = {k: v.replace('/','|') for k,v in opts.items()} # avoid path issues
        opts = [f'{k.split(".")[-1]}={v}' for k,v in opts.items()] # drop prefix
        opt_str = '_'.join(opts) # concat args
        output_dir = os.path.join(output_dir, opt_str)

    cfg.train.output_dir = output_dir
    args.output_dir = output_dir
    args.opts += [f"train.output_dir={output_dir}"]

    return cfg

def setup_wandb_logging(cfg):

    if cfg.train.use_wandb and distributed.is_main_process():
        output_dir = cfg.train.output_dir
        name = str(os.path.relpath(output_dir, os.environ['ODIR']))
        wandb.init(
            project=os.environ['WANDB_PROJECT'], 
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir = output_dir,
            resume = 'auto')

def train_setup(args):
    """
    Create configs and perform basic setups. In args should be:
        config_file: path to the config file
        output_dir: path to the output directory
    """
    load_dotenv()
    cfg = get_train_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    write_config(cfg, args.output_dir, name='pre_apply_rules_config.yaml')
    apply_scaling_rules_to_cfg(cfg)
    write_config(cfg, args.output_dir, name='config.yaml')
    setup_wandb_logging(cfg)
    logger.info(OmegaConf.to_yaml(cfg))
    return cfg