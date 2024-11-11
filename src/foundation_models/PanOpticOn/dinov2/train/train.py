# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
from fvcore.common.checkpoint import Checkpointer
import torch

from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.setup import get_autocast_dtype
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, MaskingGenerator
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import train_setup
from dinov2.utils.utils import CosineScheduler, load_pretrained_weights

from dinov2.train.ssl_meta_arch import SSLMetaArch
import time
import traceback
import shutil
from dinov2.eval.offline_train_eval import do_offline_eval


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--env", default=None, help='overwrite config with env specific vars')
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        '--fastdevrun',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Wheter to do a quick debug run with tmp directory, no wandb, small batches'
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, epoch):
    for param_group in optimizer.param_groups:
        freeze_epochs = param_group["freeze_epochs"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (0 if epoch < freeze_epochs else lr) * lr_multiplier


def save_teacher(cfg, model, iteration):
    logger.info(f'Offline Evaluation ... ')
    eval_time = time.time()
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
        logger.info(f'Saved teacher weights to {teacher_ckp_path}')
    
    eval_time = time.time() - eval_time 
    logger.info(f'Offline evaluation done ({eval_time:.2f}s)')

def do_online_eval(cfg, model, iteration, train_dataset, val_dataset, metric_logger:MetricLogger=None):
    logger.info(f'Online Evaluation ... ')
    raise NotImplementedError('Online evaluation not supported yet. Please use offline evaluation.')

    torch.cuda.synchronize()
    sleep_time = cfg.eval.eval_sleep_time
    logger.info(f'Sleeping for {sleep_time}s before evaluation...')
    time.sleep(sleep_time)
    logger.info('Done sleeping')

    autocast_dtype = get_autocast_dtype(cfg)
    output_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
    os.makedirs(output_dir, exist_ok=True)

    # process keys 
    input_keys = ['nb_knn', 'temperature', 'gather_on_cpu',  'n_per_class_list', 'n_tries']
    kwargs = {k:v for k,v in cfg.eval.items() if k in input_keys}

    logger.info(f'model datatype: {autocast_dtype}')
    start = time.time()
    model.eval()
    results = eval_knn_with_model(
        model=model,
        output_dir=output_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        autocast_dtype=autocast_dtype,
        dl_cfg=cfg.eval.dl_cfg,
        **kwargs)
    model.train()
    eval_time = time.time() - start 
    
    if metric_logger is not None and distributed.is_main_process():
        for knn_, metric_dicts in results.items():
            metric_logger.log_wandb(iteration, metric_dicts, prefix=f'val/knn/{knn_}/')
        metric_logger.log_wandb(iteration, {'eval_time': eval_time})

    logger.info(f'Online Evaluation done ({eval_time:.2f}s)')

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    use_online_eval = cfg.eval.use_online_eval

    # setup datasets
    dino_augm_cfg = cfg.train.dino_augm
    dataset = make_dataset(cfg.train.dataset, dino_augm_cfg=dino_augm_cfg, seed=cfg.train.seed)
    model.global_crops_number = dino_augm_cfg.global_crops_number
    model.local_crops_number = dino_augm_cfg.local_crops_number
    
    if use_online_eval and cfg.eval.eval_period_epoch > 0:
        logger.info('Building evaluation datasets...')
        eval_train_dataset = make_dataset(cfg.eval.train_dataset, seed=cfg.train.seed)
        eval_val_dataset = make_dataset(cfg.eval.val_dataset, seed=cfg.train.seed)

    if cfg.train.OFFICIAL_EPOCH_LENGTH == -1:
        cfg.train.OFFICIAL_EPOCH_LENGTH = math.ceil(
            len(dataset) / (cfg.train.batch_size_per_gpu * distributed.get_global_size()))
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
    ) = build_schedulers(cfg)

    # setup checkpointer

    checkpointer = Checkpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=math.ceil(cfg.train.saveckp_freq * OFFICIAL_EPOCH_LENGTH),
        max_iter=max_iter,
        max_to_keep=1,
    )

    # setup data loader
    if isinstance(dino_augm_cfg.global_crops_size, int):
        img_size = dino_augm_cfg.global_crops_size
    else:
        img_size = tuple(dino_augm_cfg.global_crops_size)[1]
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
        # global_smask_absolute_tuple=cfg.train.global_smask_absolute_tuple,
        # local_smask_absolute_tuple=cfg.train.local_smask_absolute_tuple,
    )

    sampler_type = SamplerType.INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        drop_last=cfg.train.drop_last,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        collate_fn=collate_fn,
    )

    # setup training loop

    iteration = start_iter
    epoch = iteration // OFFICIAL_EPOCH_LENGTH
    eff_bsz = cfg.train.batch_size_per_gpu * distributed.get_global_size()
    logger.info(    f'#dataset={len(dataset)}, ' + \
                    f'eff_bsz={eff_bsz}, ' + \
                    f'OFFICIAL_EPOCH_LENGTH={OFFICIAL_EPOCH_LENGTH}')
    logger.info(f"Starting training from: iteration={iteration}, epoch={epoch}\n")
    metric_logger = MetricLogger(delimiter="  ", 
                                 output_dir=cfg.train.output_dir, 
                                 output_file = 'training_metrics.json',
                                 use_wandb=cfg.train.use_wandb)
    header = "Training"

    period_epoch = cfg.eval.eval_period_epoch
    period_iter = cfg.eval.eval_period_iterations
    if period_epoch > 0 and period_iter > 0:
        raise ValueError("Only one of period_epoch or period_iter can be set.")
    elif period_iter < 0 and period_epoch < 0:
        raise ValueError("Either period_epoch or period_iter must be set.")
    elif period_epoch > 0:
        eval_period_iterations = period_epoch * OFFICIAL_EPOCH_LENGTH
    else: 
        eval_period_iterations = period_iter

    # training loop
 
    for data in metric_logger.log_every(
        data_loader,
        cfg.train.log_every_n_steps,
        header,
        max_iter,
        start_iter,
        use_self_timemeters=True,
        epoch_len = OFFICIAL_EPOCH_LENGTH,
        nsamples_per_iter=eff_bsz,
        dataset_len=len(dataset),
    ):
        current_batch_size = data["collated_global_crops"]['imgs'].shape[0] / 2
        if iteration > max_iter:
            print('Max iterations reached. Exiting training loop.')
            break

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, epoch)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(teacher_temp=teacher_temp)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing & testing

        if distributed.is_main_process(): # this only works with DDP not FSDP
            periodic_checkpointer.step(iteration)
        if eval_period_iterations > 0 and (iteration + 1) % eval_period_iterations == 0:
            if use_online_eval:
                do_online_eval(cfg, 
                    model.teacher['backbone'], 
                    iteration, 
                    eval_train_dataset, 
                    eval_val_dataset, 
                    metric_logger=metric_logger)
            else:
                save_teacher(cfg, model, f"{iteration}")
            torch.cuda.synchronize()
        
        iteration = iteration + 1


    metric_logger.synchronize_between_processes()
    ret = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # do offline evaluation
    if not use_online_eval \
        and (os.path.exists(os.path.join(cfg.train.output_dir, 'eval')) \
            or cfg.eval.include_final_ckpt):
        logger.info('Offline Evaluation at end of training ... ')
        
        # clear cuda memory
        del model, optimizer, data_loader, dataset, checkpointer, periodic_checkpointer
        torch.cuda.empty_cache()

        # start eval
        is_main = distributed.is_main_process()
        result_dict = do_offline_eval(cfg.train.output_dir, 
                                      cfg.eval.config_dir, 
                                      remove_ckpts=cfg.eval.remove_ckpts,
                                      return_all_res=cfg.eval.return_all_results,
                                      recompute=cfg.eval.recompute,
                                      include_final_ckpt=cfg.eval.include_final_ckpt,
                                      overwrites=cfg.eval.overwrites)

        if cfg.train.use_wandb and is_main:

            for res in result_dict:
                iteration = int(res.pop('iteration'))

                # log first metric separately for better viz in wandb
                prev_task = ''
                for k,v in res.items():
                    curr_task = '/'.join(k.split('/')[:-1])
                    if curr_task != prev_task:
                        prefix = 'val'
                        prev_task = curr_task
                    else:
                        prefix = 'val_all'
                    log_dict = {k:v}
                    metric_logger.log_wandb(iteration, log_dict, log_step=False, prefix=prefix)
    else:
        logger.info('No offline evaluation done at end of training')
    return ret


def main(args):
    cfg = train_setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    if len(cfg.MODEL.pretrained_weights) > 0:
        logger.info('Loading pretrained weights for SSLMetaArch ...')   
        load_pretrained_weights(model, cfg.MODEL.pretrained_weights)
    model.prepare_for_distributed_training()
    logger.info(f'model weight datatype: {model.teacher["backbone"].pos_embed.dtype}')
    logger.info("Model:\n{}".format(model))

    try: 
        do_train(cfg, model, resume=not args.no_resume)
    except Exception as e:
        logger.error('Error message: ' + str(e))
        logger.error('Full traceback:\n ' + traceback.format_exc())
        # logger.error(e, stack_info=True, exc_info=True)
        raise e
    finally:
        if args.fastdevrun:
            shutil.rmtree(cfg.train.output_dir)
            logger.info('Debug run complete. Removed directory.')
            return

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)