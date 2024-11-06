from .SatMAE.models_vit_group_channels_seg import vit_large_patch16 as vit_large_patch16_seg
from .SatMAE.models_vit_group_channels import vit_large_patch16 as vit_large_patch16_cls

import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
import pdb
from util.misc import resize

# upernet + mae from mmsegmentation
class UperNet(torch.nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super(UperNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head
    
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a

class SatMAE(nn.Module):
    def __init__(self, config):
        super(SatMAE, self).__init__()

        #get the params for the model
        kwargs = {}
        kwargs['img_size'] = config.image_resolution
        kwargs['patch_size'] = config.patch_size
        kwargs['in_chans'] = config.num_channels
        kwargs['channel_groups'] = config.channel_groups
        
        self.encoder = vit_large_patch16_seg(**kwargs)
        #Load pretrained weights
        checkpoint = torch.load(config.pretrained_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
        logger.debug(msg)
        
        self.out_features = config.out_features
        self.model = self.encoder
        self.task = config.task

        if config.freeze_backbone:
            self.freeze(self.encoder)

        if config.task == 'classification':
            #add linear layer
            pass

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

    def params_to_optimize(self):
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
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
        match self.task:
            case 'classification':
                raise NotImplementedError("on going")
            case 'segmentation':
                out, out_aux =  self.seg_model(samples)
                return out, out_aux




