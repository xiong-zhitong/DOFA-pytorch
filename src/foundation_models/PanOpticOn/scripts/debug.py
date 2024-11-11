import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import datetime
import random
from functools import partial
import logging
import functools
from PanOpticOn.dinov2.eval.dist_task_manager import _work_knn, setup_logger
from dinov2.data.loaders import make_dataset
from dinov2.eval.knn import eval_knn_with_model
from dinov2.eval.setup import get_autocast_dtype
from dinov2.models import build_model_from_cfg
import dinov2.distributed as distributed
from copy import deepcopy
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
import os
import logging
import sys

# base_dir = '/data/panopticon/logs/dino_logs/debug/7/eval/59'
# cfg = OmegaConf.load('/data/panopticon/logs/dino_logs/debug/7/config.yaml')


# distributed.enable(overwrite=True)
# _work_knn(base_dir, cfg)

# train_dataset = make_dataset(cfg.evaluation.train_dataset, seed=cfg.train.seed)
# dl = DataLoader(train_dataset, batch_size=5)

# # val_dataset = make_dataset(cfg.evaluation.val_dataset, seed=cfg.train.seed)
# model, _ = build_model_from_cfg(cfg, only_teacher=True)
# model.eval()

# device = torch.device('cuda')
# model.to(device)
# autocast_dtype = get_autocast_dtype(cfg)



# with torch.cuda.amp.autocast(dtype=autocast_dtype):

#     for i, batch in enumerate(dl):
#         print(i)
#         if i == 2:
#             break
#         x_dict = {k: v.to(device) for k, v in batch[0].items()}
#         out = model(x_dict)


# import logging
# import sys

# logger = logging.getLogger("dinov2")
# logger.setLevel(logging.DEBUG)

# # formatter
# fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
# fmt_message = "%(message)s"
# fmt = fmt_prefix + fmt_message
# datefmt = "%Y%m%d %H:%M:%S"
# formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

# # sysout
# handler = logging.StreamHandler(stream=sys.stdout)
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# # fileout
# filename = '/home/lewaldm/code/PanOpticOn/scripts/log'
# handler = logging.StreamHandler(open(filename, "a"))
# handler.setLevel(logging.DEBUG)
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# logger.info('initialized')

# # try:
# #     asdf
# # except:
# #     logging.exception('failed')
# #     raise
# # raise NotImplementedError('err')
# # logger.info('test2')



# try:
#     asdf
# except Exception as e:
#     logging.error(e, exc_info=True)


# import traceback

# def fun():
#     raise RuntimeError('runtime')

# def fun2():
#     try:
#         fun()
#     except Exception as e:
#         raise ValueError('value') from e
    
# try:
#     fun2()
# except Exception as e:
#     print(traceback.format_exc())
#     print(e)


# debug fmow dataset
# from omegaconf import OmegaConf
# from dinov2.data.loaders import make_dataset 
# import logging

# logger = logging.getLogger("dinov2")

# cfg = OmegaConf.load('/home/lewaldm/code/PanOpticOn/dinov2/configs/sweep/lr_det_smasks.yaml')
# ds = make_dataset(cfg.train.dataset)
# ds.df.at[0, 'path'] = [p + 'fake' if i %4 == 0 else p for i,p in enumerate(ds.df.iloc[0]['path']) ]
# print(ds.df.iloc[0]['path'])

# for i in range(10):
#     print(i)
#     ds[0]
# print('done')




# debug knn
# distributed.enable(overwrite=True)

run_dir = '/data/panopticon/logs/dino_logs/simplattn/long/lr=1e-3_warmup=0/'
# run_dir = '/data/panopticon/logs/dino_logs/simplattn/s_lr/lr=1e-3_warmup=0_fw=backbone=0_inp=false'
cfg = OmegaConf.load(os.path.join(run_dir, 'config.yaml'))
eval_dir = os.path.join(run_dir, 'knn_final_eval')

global logger
os.makedirs(eval_dir, exist_ok=True)
logger = setup_logger('dinov2', os.path.join(eval_dir, 'log'), to_sysout=True)

# config equals train_config.evaluation
train_dataset = make_dataset(cfg.evaluation.train_dataset, seed=cfg.train.seed)
val_dataset = make_dataset(cfg.evaluation.val_dataset, seed=cfg.train.seed)

# load model with correct weights
cfg.student.pretrained_weights = [OmegaConf.create(dict(
    path=os.path.join(run_dir, 'model_final.pth'),
    checkpoint_key='model',
    include='teacher.backbone.',
    prefix_load_map={'teacher.backbone.': ''},))]
model, _ = build_model_from_cfg(cfg, only_teacher=True)
model.eval()
model.cuda()

autocast_dtype = get_autocast_dtype(cfg)

# extract kwargs
input_keys = ['nb_knn', 'temperature', 'gather_on_cpu',  'n_per_class_list', 'n_tries']
kwargs = {k:v for k,v in cfg.evaluation.items() if k in input_keys} 

# actual knn
eval_time = time.time()
results = eval_knn_with_model(
    model=model,
    output_dir=eval_dir,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    autocast_dtype=autocast_dtype,
    dl_cfg=cfg.evaluation.dl_cfg,
    **kwargs)

logger.info(f'Online Evaluation done ({time.time() - eval_time:.2f}s)')