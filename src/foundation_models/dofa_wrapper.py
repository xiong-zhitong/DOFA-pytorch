import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from util.misc import resize
import torch.nn.functional as F
from util.misc import seg_metric, cls_metric

from .DOFA.models_dwv_seg import vit_base_patch16 as vit_base_patch16_seg
from .DOFA.models_dwv_seg import vit_large_patch16 as vit_large_patch16_seg
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls

class UperNet(nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head
    
    def forward(self, x_dict):
        x = x_dict['imgs']
        feat = self.backbone.forward_features(x, x_dict['wavelengths'])
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a

class LightningDOFA(pl.LightningModule):
    def __init__(self, config, data_config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.data_config = data_config
        self.warmup_epochs = 3
        
        # Initialize encoder based on task
        self.encoder = None
        if config.task == "classification":
            self.encoder = (vit_base_patch16_cls(num_classes=config.num_classes) 
                          if config.dofa_size == 'dofa_base'
                          else vit_large_patch16_cls(num_classes=config.num_classes))
        else:  # segmentation
            self.encoder = (vit_base_patch16_seg(drop_path_rate=0,) 
                          if config.dofa_size == 'dofa_base'
                          else vit_large_patch16_seg(drop_path_rate=0,))
        
        # Load pretrained weights
        check_point = torch.load(config.pretrained_path)
        self.encoder.load_state_dict(check_point, strict=False)
        
        self.out_features = config.out_features
        self.task = config.task
        
        if config.freeze_backbone:
            self.freeze(self.encoder)
            
        # Task-specific setup
        if config.task == 'classification':
            trunc_normal_(self.encoder.head.weight, std=0.01)
            self.encoder.head = nn.Sequential(
                nn.BatchNorm1d(self.encoder.head.in_features, affine=False, eps=1e-6),
                self.encoder.head
            )
            self.unfreeze(self.encoder.head)
            
            # Classification specific loss
            self.criterion = (nn.MultiLabelSoftMarginLoss() if config.multilabel 
                            else nn.CrossEntropyLoss())
            
        elif config.task == 'segmentation':
            edim = config.embed_dim
            self.neck = Feature2Pyramid(
                embed_dim=edim,
                rescales=[4, 2, 1, 0.5],
            )
            self.decoder = UPerHead(
                in_channels=[edim, edim, edim, edim],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=512,
                dropout_ratio=0.1,
                num_classes=config.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            )
            self.aux_head = FCNHead(
                in_channels=edim,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=config.num_classes,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            )
            
            self.seg_model = UperNet(self.encoder, self.neck, self.decoder, self.aux_head)
            
            # Segmentation specific losses
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_aux = nn.CrossEntropyLoss()
    
    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
            
    def forward(self, samples):
        x_dict = {
            'imgs': samples,
            'wavelengths': self.config.band_wavelengths
        }
        
        if self.task == 'classification':
            out_logits, feats = self.encoder(samples, self.config.band_wavelengths)
            return (out_logits, feats) if self.out_features else out_logits
        else:  # segmentation
            return self.seg_model(x_dict)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        
        if self.task == 'classification':
            outputs = self(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.criterion(outputs, targets)
            
            # Calculate accuracy
            acc1, acc5 = cls_metric(self.data_config, outputs, targets)
            self.log('train_loss', loss.to(self.device), on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc1', acc1.to(self.device), on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc5', acc5.to(self.device), on_step=True, on_epoch=True, prog_bar=True)
            
        else:  # segmentation
            outputs, aux_outputs = self(images)
            loss = self.criterion(outputs, targets) + self.criterion_aux(aux_outputs, targets)
            
            # Calculate mIoU
            miou, acc = seg_metric(self.data_config, outputs, targets)
            self.log('train_loss', loss.to(self.device), on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_miou', miou.to(self.device), on_step=True, on_epoch=True, prog_bar=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        
        if self.task == 'classification':
            outputs = self(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.criterion(outputs, targets)
            
            acc1, acc5 = cls_metric(self.data_config, outputs, targets)
            self.log('val_loss', loss.to(self.device), on_epoch=True, sync_dist=True)
            self.log('val_acc1', acc1.to(self.device), on_epoch=True, sync_dist=True)
            self.log('val_acc5', acc5.to(self.device), on_epoch=True, sync_dist=True)
            
        else:  # segmentation
            outputs, aux_outputs = self(images)
            loss = self.criterion(outputs, targets) + self.criterion_aux(aux_outputs, targets)
            
            miou, acc = seg_metric(self.data_config, outputs, targets)
            self.log('val_loss', loss.to(self.device), on_epoch=True, sync_dist=True)
            self.log('val_miou', miou.to(self.device), on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        if self.task == 'classification':
            from util.lars import LARS
            optimizer = LARS(self.encoder.head.parameters(), 
                           lr=self.config.lr,
                           weight_decay=self.config.weight_decay)
        else:
            param_groups = [
                {'params': self.neck.parameters(), 'lr': 0.001},
                {'params': self.decoder.parameters(), 'lr': 0.001},
                {'params': self.aux_head.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.AdamW(param_groups)
            
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda epoch: epoch / self.warmup_epochs if epoch < self.warmup_epochs else 1.0)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs-self.warmup_epochs,
            eta_min=0.0001,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.warmup_epochs])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    