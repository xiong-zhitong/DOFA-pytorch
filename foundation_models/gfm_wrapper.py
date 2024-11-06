from .GFM import build_swin_cls, build_swin_seg
import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
import pdb
from util.misc import resize

# upernet + mae from mmsegmentation
class UperNet(torch.nn.Module):
    def __init__(self, backbone, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.aux_head = aux_head
    
    def forward(self, x):
        feat = self.backbone(x)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a


class GFM(nn.Module):
    def __init__(self, config):
        super(GFM, self).__init__()

        self.encoder = build_swin_seg(config) # checkpoint already loaded
        
        self.out_features = config.out_features
        self.model = self.encoder
        self.task = config.task

        if config.freeze_backbone:
            self.freeze(self.encoder)

        if config.task == 'classification':
            raise NotImplementedError("on going")

        elif config.task == 'segmentation':
            # create model: upernet + mae
            edim = config.embed_dim
            self.decoder = UPerHead(
                in_channels=[256, 512, 1024, 1024],
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

            self.seg_model = UperNet(self.encoder, self.decoder, self.aux_head)

    def params_to_optimize(self):
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
            case 'segmentation':
                parameters_to_optimize = (list(self.decoder.parameters()) + \
                        list(self.aux_head.parameters()))
                return parameters_to_optimize

    def check_requires_grad(self, module):
        return all(param.requires_grad for param in module.parameters())

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, samples):
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
            case 'segmentation':
                out, out_aux =  self.seg_model(samples)
                return out, out_aux




