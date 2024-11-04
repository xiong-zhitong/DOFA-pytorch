from .GFM import build_swin_cls, build_swin_seg
import torch.nn as nn
import torch
# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from loguru import logger
import pdb
import torch.nn.functional as F

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
        out = self.resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out_a = self.aux_head(feat)
        out_a = self.resize(out_a, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, out_a

    @staticmethod
    def resize(input,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            warning=True):
        if warning:
            if size is not None and align_corners:
                input_h, input_w = tuple(int(x) for x in input.shape[2:])
                output_h, output_w = tuple(int(x) for x in size)
                if output_h > input_h or output_w > output_h:
                    if ((output_h > 1 and output_w > 1 and input_h > 1
                        and input_w > 1) and (output_h - 1) % (input_h - 1)
                            and (output_w - 1) % (input_w - 1)):
                        warnings.warn(
                            f'When align_corners={align_corners}, ')
        return F.interpolate(input, size, scale_factor, mode, align_corners)


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
            #add linear layer
            pass

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
                return self.encoder.GAP_FFN_s2[1].parameters()
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
                all_output = self.model(optical_images=samples)
                out_logits = all_output['optical_GAP']
                out_feats = all_output['optical_encodings']
                if self.out_features:
                    return out_logits, out_feats
                else:
                    return out_logits
            case 'segmentation':
                out, out_aux =  self.seg_model(samples)
                return out, out_aux




