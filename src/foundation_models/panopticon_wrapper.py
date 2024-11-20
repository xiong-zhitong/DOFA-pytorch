import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
# support for object detection tasks
# mmdet

from loguru import logger
import pdb
from util.misc import resize

import sys
import os
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/PanOpticOn")

from dinov2.eval.setup import parse_model_obj
from dinov2.utils.data import load_ds_cfg, extract_wavemus
import math
from einops import rearrange
from .lightning_task import LightningTask
from einops import rearrange
from util.misc import seg_metric, cls_metric

class PanapticonClassification(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)
        
        model_folder = config.pretrained_path
        self.encoder = parse_model_obj(model_obj=model_folder, return_with_wrapper=False)
        # Loaded pretrained weights

        if config.freeze_backbone:
            self.freeze(self.encoder)
        self.linear_classifier = nn.Linear(config.embed_dim, config.num_classes)

        self.criterion = (nn.MultiLabelSoftMarginLoss() if config.multilabel 
                          else nn.CrossEntropyLoss())
        
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)
    
    def forward(self, samples):
        x_dict = {}
        BSIZE = samples.shape[0]
        x_dict['imgs'] = samples
        chn_ids = extract_wavemus(load_ds_cfg(self.config.ds_name), return_sigmas=self.config.full_spectra)
        x_dict['chn_ids'] = torch.tensor(chn_ids, dtype=torch.long, device=samples.device).unsqueeze(0).repeat([BSIZE,1])

        out = self.encoder.forward_features(x_dict)
        global_pooled = out["x_norm_patchtokens"].mean(dim=1)
        out_logits = self.linear_classifier(global_pooled)
        return out_logits, global_pooled

    def params_to_optimize(self):
        return self.linear_classifier.parameters()
    
    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(f'{prefix}_loss', self.loss(outputs, targets), on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_acc1', acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_acc5', acc5, on_step=True, on_epoch=True, prog_bar=True)



class PanapticonSegmentation(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)
        self.encoder = torch.hub.load('facebookresearch/dinov2', config.dino_size)
        if config.freeze_backbone:
            self.freeze(self.encoder)
        
        edim = config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4, in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
            channels=512, dropout_ratio=0.1, num_classes=config.num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )
        self.aux_head = FCNHead(
            in_channels=edim, in_index=2, channels=256, num_convs=1, concat_input=False,
            dropout_ratio=0.1, num_classes=config.num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(outputs[1], labels)

    def forward(self, samples):
        x_dict = {}
        BSIZE = samples.shape[0]
        x_dict['imgs'] = samples
        chn_ids = extract_wavemus(load_ds_cfg(self.config.ds_name), return_sigmas=self.config.full_spectra)
        x_dict['chn_ids'] = torch.tensor(chn_ids, dtype=torch.long, device=samples.device).unsqueeze(0).repeat([BSIZE,1])

        outputs = self.encoder.get_intermediate_layers(x_dict, [4, 6, 10, 11])
        feats = [rearrange(out, "n (h w) c -> n c h w", h=int(out.size(1)**0.5)) for out in outputs]
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(out_a, size=samples.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a

    def params_to_optimize(self):
        return list(self.neck.parameters()) + list(self.decoder.parameters()) + list(self.aux_head.parameters())

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_miou', miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{prefix}_acc', acc, on_step=True, on_epoch=True, prog_bar=True)



# Model factory for different dinov2 tasks
def PanapticonModel(args, config, data_config):
    if args.task == "classification":
        return PanapticonClassification(args, config, data_config)
    elif args.task == "segmentation":
        return PanapticonSegmentation(args, config, data_config)
    else:
        raise NotImplementedError("Task not supported")