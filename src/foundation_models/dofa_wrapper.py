import torch
import os
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
from torchvision.datasets.utils import download_url
from peft import LoraConfig, get_peft_model


class DofaClassification(LightningTask):

    url = "https://huggingface.co/earthflow/dofa/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.lora = model_config.get("lora", False)

        self.full_finetune = model_config.get("full_finetune", False)

        # can only be one of the two
        assert not (self.lora and self.full_finetune), "Can only use one of LoRA or full finetune bot not both to true"

        self.encoder = (
            vit_base_patch16_cls(num_classes=data_config.num_classes) if model_config.size == "large" \
                else vit_base_patch16_cls(num_classes=data_config.num_classes)
            )

        print(self.encoder)


        # look for pretrained weights
        dir = os.getenv("MODEL_WEIGHTS_DIR")
        filename = model_config.pretrained_path
        path = os.path.join(dir, filename)
        if not os.path.exists(path):
            # download the weights from HF
            download_url(self.url.format(filename), dir, filename=filename)

        # Load pretrained weights
        check_point = torch.load(path)
        self.encoder.load_state_dict(check_point, strict=False)

        if self.lora and model_config.lora:
            self.apply_peft(self.encoder, lora_cfg=model_config.lora)

        if model_config.freeze_backbone:
            if self.lora:
                # TODO not implemented yet I think
                self.freeze_non_lora_params(self.encoder)
            else:
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

        self.model_config = model_config
        self.data_config = data_config

    def freeze_non_lora_params(self, encoder):
        raise NotImplementedError("Not implemented yet: CANNOT freeze non-LoRA parameters")

    def apply_peft(self, encoder, lora_cfg: dict):
        """
        Apply LoRA to the last few layers of the encoder using PEFT.
        """

        print("LORA: Applying PEFT: ", lora_cfg)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_rank", 16),  # Rank of LoRA
            lora_alpha=lora_cfg.get("lora_alpha", 16),  # Scaling factor for LoRA
            target_modules=cfg.get("lora_target_modules",  "blocks.*.attn.qkv"), #["qkv", "proj"]
            lora_dropout=lora_cfg.get("lora_dropout", 0.),  # Dropout rate for LoRA
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("lora_task_type", None)  # Task type (use appropriate type for your model), "SEQ_CLS"
        )

        # Wrap the encoder with PEFT
        self.encoder = get_peft_model(encoder, peft_config)


    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        out_logits, feats = self.encoder(samples, self.data_config.band_wavelengths)
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        if self.lora:
            # Include LoRA parameters for optimization
            lora_params = [p for n, p in self.encoder.named_parameters() if "lora" in n]
            return list(self.encoder.head.parameters()) + lora_params
        if self.full_finetune:
            return list(self.encoder.parameters())
        else:
            return list(self.encoder.head.parameters())

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
            if model_config.size == "base"
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
