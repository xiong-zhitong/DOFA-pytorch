import pytorch_lightning as pl
import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
from util.misc import resize
import math
from einops import rearrange
from util.misc import seg_metric, cls_metric


# upernet + mae from mmsegmentation
class UperNet(torch.nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head
        self.idx_blocks_to_return = [4, 6, 10, 11]
    
    def forward(self, x_dict):  
        outputs = self.backbone.get_intermediate_layers(x_dict, self.idx_blocks_to_return)
        x = x_dict['imgs']
        N,HW,C = outputs[0].shape
        H = W = int(math.sqrt(HW))
        feat = [rearrange(out, "n (h w) c -> n c h w", h=H, w=W) for out in outputs]
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a


class LightningDinov2(pl.LightningModule):
    def __init__(self, args, config, data_config):
        super(LightningDinov2, self).__init__()

        self.config = config
        self.data_config = data_config
        self.args = args
        self.encoder = torch.hub.load('facebookresearch/dinov2', config.dino_size)

        self.out_features = config.out_features
        self.model = self.encoder
        self.task = config.task

        if config.freeze_backbone:
            self.freeze(self.encoder)

        if config.task == 'classification':
            self.linear_classifier = torch.nn.Linear(config.embed_dim, config.num_classes)
            # Classification specific loss
            self.criterion = (nn.MultiLabelSoftMarginLoss() if config.multilabel 
                            else nn.CrossEntropyLoss())

        elif config.task == 'segmentation':
            # create model: upernet + mae
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

            self.criterion = nn.CrossEntropyLoss()
            self.criterion_aux = nn.CrossEntropyLoss()

    def params_to_optimize(self):
        match self.task:
            case 'classification':
                return self.linear_classifier.parameters()
            case 'segmentation':
                parameters_to_optimize = (list(self.neck.parameters()) + list(self.decoder.parameters()) + \
                        list(self.aux_head.parameters()))
                return parameters_to_optimize

    def check_requires_grad(self, module):
        return all(param.requires_grad for param in module.parameters())

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, samples):
        x_dict = {}
        x_dict['imgs'] = samples
        match self.task:
            case 'classification':
                #TODO: add cls token support; key is: x_norm_clstoken
                out = self.encoder.forward_features(x_dict)
                global_pooled = out["x_norm_patchtokens"].mean(dim=1)
                out_logits = self.linear_classifier(global_pooled)
                if self.out_features:
                    return out_logits, global_pooled
                return out_logits
            case 'segmentation':
                #logger.debug(f'dinov2: {self.config.dino_size}')
                out, out_aux =  self.seg_model(x_dict)
                return out, out_aux
            
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
            optimizer = torch.optim.SGD(self.params_to_optimize(),
                           lr=self.args.lr,
                           weight_decay=self.args.weight_decay)
        else:
            param_groups = [
                {'params': self.neck.parameters(), 'lr': 0.001},
                {'params': self.decoder.parameters(), 'lr': 0.001},
                {'params': self.aux_head.parameters(), 'lr': 0.001}
            ]
            optimizer = torch.optim.AdamW(param_groups)
            
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda epoch: epoch / self.args.warmup_epochs if epoch < self.args.warmup_epochs else 1.0)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs-self.args.warmup_epochs,
            eta_min=0.0001,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.args.warmup_epochs])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    




