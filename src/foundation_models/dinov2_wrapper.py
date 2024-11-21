import pytorch_lightning as pl
import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
from util.misc import resize
from .lightning_task import LightningTask
from einops import rearrange
from util.misc import seg_metric, cls_metric
# MMDetection imports
from mmdet.models import BACKBONES, build_backbone
from mmdet.models.dense_heads import YOLOHead, RetinaHead  # You can choose different heads
from mmdet.models.builder import build_head
from mmdet.models.necks import FPN


class DinoV2Classification(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)
        
        self.encoder = torch.hub.load('facebookresearch/dinov2', config.dino_size)
        if config.freeze_backbone:
            self.freeze(self.encoder)
        self.linear_classifier = nn.Linear(config.embed_dim, config.num_classes)
        self.criterion = (nn.MultiLabelSoftMarginLoss() if config.multilabel 
                          else nn.CrossEntropyLoss())
        
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)
    
    def forward(self, samples):
        x_dict = {"imgs": samples}
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



class DinoV2Segmentation(LightningTask):
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
        x_dict = {"imgs":samples}
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


class DinoV2ObjectDetection(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)
        self.encoder = torch.hub.load('facebookresearch/dinov2', config.dino_size)
        if config.freeze_backbone:
            self.freeze(self.encoder)
        # Neck configuration (Feature Pyramid Network)
        neck_cfg = dict(
            type='FPN',
            in_channels=[backbone_cfg['embed_dim']] * 4,  # Use Dinov2 embedding dim
            out_channels=256,
            num_outs=5
        )
        
        # Head configuration (example with RetinaHead)
        head_cfg = dict(
            type='RetinaHead',
            num_classes=config.num_classes,
            in_channels=256,
            stacked_convs=3,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]
            ),
            loss_cls=dict(
                type='FocalLoss', 
                use_sigmoid=True, 
                gamma=2.0, 
                alpha=0.25
            ),
            loss_bbox=dict(
                type='IoULoss'
            )
        )
        
        # Build model components
        self.neck = FPN(**neck_cfg)
        self.head = build_head(head_cfg)
        
        self.config = config
        self.data_config = data_config

    def forward(self, x):
        # Extract features from backbone
        x = self.encoder(x)
        # Apply neck (FPN)
        x = self.neck(x)
        
        # Get detection results
        return self.head(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        x = self.encoder(imgs)
        x = self.neck(x)
        # Compute losses
        loss_inputs = self.head.forward_train(x, targets)
        
        # Log losses
        loss_dict = self.head.loss(*loss_inputs)
        
        # Log individual losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Return total loss
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        
        # Extract features
        x = self.encoder(imgs)
        x = self.neck(x)
        # Compute losses
        loss_inputs = self.head.forward_train(x, targets)
        # Compute loss dict
        loss_dict = self.head.loss(*loss_inputs)
        # Log individual losses
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Return total loss
        return loss_dict['loss']
    
    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        
        # Extract features
        x = self.encoder(imgs)
        x = self.neck(x)
        # Compute losses
        loss_inputs = self.head.forward_train(x, targets)
        # Compute loss dict
        loss_dict = self.head.loss(*loss_inputs)
        # Log individual losses
        for key, value in loss_dict.items():
            self.log(f'test_{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Return total loss
        return loss_dict['loss']

    def params_to_optimize(self):
        return list(self.neck.parameters()) + list(self.decoder.parameters())


# Model factory for different dinov2 tasks
def DinoV2Model(args, config, data_config):
    if args.task == "classification":
        return DinoV2Classification(args, config, data_config)
    elif args.task == "segmentation":
        return DinoV2Segmentation(args, config, data_config)
    elif args.task == "object_detection":
        return DinoV2ObjectDetection(args, config, data_config)
    else:
        raise NotImplementedError("Task not supported")
