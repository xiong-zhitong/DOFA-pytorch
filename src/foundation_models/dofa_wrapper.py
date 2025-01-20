import torch
import torch.nn as nn
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls
from .DOFA.models_dwv_seg import vit_base_patch16 as vit_base_patch16_seg
from .DOFA.models_dwv_seg import vit_large_patch16 as vit_large_patch16_seg
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from util.misc import resize
from .lightning_task import LightningTask
from timm.models.layers import trunc_normal_
from util.misc import seg_metric, cls_metric
from huggingface_hub import hf_hub_download


class DofaClassification(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = (
            vit_base_patch16_cls(num_classes=data_config.num_classes)
            if model_config.dofa_size == "dofa_base"
            else vit_large_patch16_cls(num_classes=data_config.num_classes)
        )

        # look for pretrained weights
        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            hf_hub_download(
                repo_id="earthflow/dofa",
                filename=filename,
                cache_dir=dir,
                local_dir=dir,
            )

        # Load pretrained weights
        check_point = torch.load(path)
        self.encoder.load_state_dict(check_point, strict=False)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        trunc_normal_(self.encoder.head.weight, std=0.01)
        self.encoder.head = nn.Sequential(
            nn.BatchNorm1d(self.encoder.head.in_features, affine=False, eps=1e-6),
            self.encoder.head,
        )
        self.unfreeze(self.encoder.head)

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        out_logits, feats = self.encoder(samples, self.data_config.band_wavelengths)
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        return self.encoder.head.parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)


class DofaSegmentation(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.encoder = (
            vit_base_patch16_seg(
                drop_path_rate=0,
            )
            if model_config.dofa_size == "dofa_base"
            else vit_large_patch16_seg(
                drop_path_rate=0,
            )
        )

        # Load pretrained weights
        check_point = torch.load(model_config.pretrained_path)
        self.encoder.load_state_dict(check_point, strict=False)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
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
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples):
        feats = self.encoder(samples, self.data_config.band_wavelengths)
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
def DofaModel(args, model_config, data_config):
    if args.task == "classification":
        return DofaClassification(args, model_config, data_config)
    elif args.task == "segmentation":
        return DofaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
