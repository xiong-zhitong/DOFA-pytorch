from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from .GFM import build_swin_cls, build_swin_seg
import torch.nn as nn

# use mmsegmentation for upernet+mae
from mmseg.models.decode_heads import UPerHead, FCNHead

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from ..util.misc import resize

from .base import LinearHead


def load_encoder(model_config, data_config):
    model_config.num_classes = data_config.num_classes
    encoder = build_swin_cls(model_config)
    return encoder


class GFMClsReg(LightningClsRegTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        # trunc_normal_(self.encoder.head.weight, std=0.01)
        self.linear_classifier = LinearHead(
            in_features=self.encoder.head.in_features,
            num_classes=data_config.num_classes,
        )
        self.encoder.head = self.linear_classifier
        self.dot_str_of_linear_classifier = 'head'
        trunc_normal_(self.linear_classifier.head.head[1].weight, std=0.01)

        self.freeze_and_return_params()


    def forward(self, x):
        x, _ = self.encoder(x)
        return x



class GFMSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        edim = model_config.embed_dim
        self.decoder = UPerHead(
            in_channels=[256, 512, 1024, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )

        self.freeze_and_return_params()

    def forward(self, samples):
        # overwrites parent method since GFM has no neck
        feats = self.encoder(samples)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def freeze_and_return_params(self):
        # overwrites parent method since GFM has no neck
        segm_params = (
            # list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters()))

        if self.training_mode == 'full_finetune':
            self.unfreeze(self.encoder)
            self.unfreeze(segm_params)
        elif self.training_mode == 'frozen_backbone':
            self.freeze(self.encoder)
            self.unfreeze(segm_params)
        else:
            raise ValueError(f"Invalid mode: {self.training_mode}")

        return segm_params



# Model factory for different dinov2 tasks
def GFMModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return GFMClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return GFMSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
