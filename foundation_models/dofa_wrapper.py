from .DOFA.models_dwv_seg import vit_base_patch16 as vit_base_patch16_seg
from .DOFA.models_dwv_seg import vit_large_patch16 as vit_large_patch16_seg
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
from util.misc import resize
import pdb
from einops import rearrange

# upernet + mae from mmsegmentation
class UperNet(torch.nn.Module):
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


class DOFA(nn.Module):
    def __init__(self, config):
        super(DOFA, self).__init__()

        self.config = config
        self.encoder = None
        match config.task:
            case "classification":
                self.encoder = vit_base_patch16_cls(num_classes=config.num_classes) if self.config.dofa_size == 'dofa_base'\
                      else vit_large_patch16_cls(num_classes=config.num_classes)
            case "segmentation":
                self.encoder = vit_base_patch16_seg(drop_path_rate=0,) if self.config.dofa_size == 'dofa_base' \
                    else vit_large_patch16_seg(drop_path_rate=0,)
        
        check_point = torch.load(config.pretrained_path)
        self.encoder.load_state_dict(check_point, strict=False)

        self.out_features = config.out_features
        #pdb.set_trace()
        self.task = config.task

        if config.freeze_backbone:
            self.freeze(self.encoder)

        if config.task == 'classification':
            trunc_normal_(self.encoder.head.weight, std=0.01)
            self.encoder.head = torch.nn.Sequential(torch.nn.BatchNorm1d(\
                self.encoder.head.in_features, affine=False, eps=1e-6), self.encoder.head)
            self.unfreeze(self.encoder.head)

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
                return self.encoder.head.parameters()
            case 'segmentation':
                parameters_to_optimize = (list(self.neck.parameters()) + list(self.decoder.parameters()) + \
                        list(self.aux_head.parameters()))
                return parameters_to_optimize

    def check_requires_grad(self, module):
        return all(param.requires_grad for param in module.parameters())

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def forward(self, samples):
        x_dict = {}
        x_dict['imgs'] = samples
        x_dict['wavelengths'] = self.config.band_wavelengths
        match self.task:
            case 'classification':
                out_logits, feats = self.encoder(samples, self.config.band_wavelengths)
                if self.out_features:
                    return out_logits, feats
                else:
                    out_logits

            case 'segmentation':
                out, out_aux =  self.seg_model(x_dict)
                return out, out_aux