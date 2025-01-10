from .lightning_task import LightningTask
from .CROMA.use_croma import PretrainedCROMA
import torch.nn as nn
import torch

# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from util.misc import resize, seg_metric, cls_metric


class CromaClassification(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)

        self.encoder = PretrainedCROMA(
            pretrained_path=config.pretrained_path,
            size=config.size,
            modality=config.modality,
            image_resolution=config.image_resolution,
        )

        # pretrained weights loaded
        if config.freeze_backbone:
            self.freeze(self.encoder)

        self.encoder.GAP_FFN_s2[1] = torch.nn.Linear(
            self.encoder.GAP_FFN_s2[1].in_features, config.num_classes
        )
        self.unfreeze(self.encoder.GAP_FFN_s2[1])
        del self.encoder.GAP_FFN_s2[2:]

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        all_output = self.encoder(optical_images=samples)
        out_logits = all_output["optical_GAP"]
        feats = all_output["optical_encodings"]
        return (out_logits, feats) if self.config.out_features else out_logits

    def params_to_optimize(self):
        return self.encoder.GAP_FFN_s2[1].parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)


class CromaSegmentation(LightningTask):
    def __init__(self, args, config, data_config):
        super().__init__(args, config, data_config)
        self.encoder = PretrainedCROMA(
            pretrained_path=config.pretrained_path,
            size=config.size,
            modality=config.modality,
            image_resolution=config.image_resolution,
        )

        # pretrained weights Loaded
        if config.freeze_backbone:
            self.freeze(self.encoder)

        edim = config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=config.num_classes,
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
            num_classes=config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples):
        feats = self.encoder(optical_images=samples)["out_feats"]
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)


# Model factory for different dinov2 tasks
def CromaModel(args, config, data_config):
    if args.task == "classification":
        return CromaClassification(args, config, data_config)
    elif args.task == "segmentation":
        return CromaSegmentation(args, config, data_config)
    else:
        raise NotImplementedError("Task not supported")
