"""Base Class."""

from lightning import LightningModule
import torch
from util.misc import resize, cls_metric, seg_metric, reg_metric
import torch.nn as nn

try:
    from mmseg.models.necks import Feature2Pyramid
    from mmseg.models.decode_heads import UPerHead, FCNHead
except:
    print("MMSEG not installed, skipping imports")


class LightningTask(LightningModule):
    def __init__(self, args, model_config, data_config):
        super().__init__()
        self.model_config = model_config  # model_config
        self.args = args  # args for optimization params
        self.data_config = data_config  # dataset_config
        self.training_mode = model_config.training_mode
        self.save_hyperparameters()

    def freeze_and_return_params(self):
        """ freeze & unfreeze weights according to self.training_mode, also
            returns all parameters to optimize """
        raise NotImplementedError('Subclass must implement this method')

    def forward(self, x):
        raise NotImplementedError('Subclass must implement this method')

    def loss(self, outputs, labels):
        raise NotImplementedError('Subclass must implement this method')

    def log_metrics(self, outputs, targets, prefix="train"):
        raise NotImplementedError('Subclass must implement this method')

    def _proc_param_obj(self, obj):
        if isinstance(obj, nn.Module): 
            params = obj.parameters()
        elif isinstance(obj, list) and len(obj[0]) == 1: # .parameters()
            params = obj
        elif isinstance(obj, list) and len(obj[0]) == 2: # .named_parameters()
            params = [p for _, p in obj]
        else:
            raise ValueError(f"Invalid object: {obj}")
        return params

    def freeze(self, obj):
        for p in self._proc_param_obj(obj):
            p.requires_grad = False

    def unfreeze(self, obj):
        for p in self._proc_param_obj(obj):
            p.requires_grad = True

    def training_step(self, batch, batch_idx):
        # current_lr = self.optimizers().param_groups[0]['lr']
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="test")
        return loss

    def configure_optimizers(self):
        if self.data_config.task in ["classification", "regression"]:
            optimizer = torch.optim.SGD(
                self.freeze_and_return_params(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.freeze_and_return_params(), lr=self.args.lr)

        world_size = self.args.num_gpus if self.args.num_gpus >= 1 else 1
        num_warmup_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.warmup_epochs
            // world_size)
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.epochs
            // world_size)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            anneal_strategy="cos",  # Cosine annealing
            pct_start=float(num_warmup_steps)
            / float(total_steps),  # Percentage of warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class LightningClsRegTask(LightningTask):

    encoder: torch.nn.Module
    linear_classifier: torch.nn.Module
    dot_str_of_linear_classifier: str  # state_dict key to the linear classifier if assigned to the encoder, else empty

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        task = data_config.task

        if task == 'classification':
            self.criterion = (
                nn.MultiLabelSoftMarginLoss()
                if self.data_config.multilabel
                else nn.CrossEntropyLoss())
            self.log_metrics = self._log_metrics_cls

        elif task == 'regression':
            self.criterion = nn.MSELoss()
            self.log_metrics = self._log_metrics_reg

        else: 
            raise NotImplementedError()
              
    def freeze_and_return_params(self):
        """ freeze / unfreeze weights & return parameters to optimize 
            according to self.training_mode"""
        mode = self.training_mode

        if mode == 'linear_probe':
            self.freeze(self.encoder)
            self.unfreeze(self.linear_classifier)

            params_to_optimize = self.linear_classifier.parameters()

        elif mode == 'partial_finetune':
            assert 'params_to_train' in self.model_config, "params_to_train not found in model_config"
            self.freeze(self.encoder)
            self.unfreeze(self.linear_classifier)
            params_to_unfreeze = self._filter_named_params(
                self._get_encoder_params_without_head(), self.model_config.params_to_train)
            self.unfreeze(params_to_unfreeze)

            params_to_optimize = list(self.linear_classifier.parameters()) + \
                list([p for _, p in params_to_unfreeze])

        elif mode == 'lora':
            self.freeze(self.encoder)
            self.unfreeze(self.linear_classifier)
            lora_params = self._filter_named_params(
                self._get_encoder_params_without_head(), ['lora'])
            assert len(lora_params) > 0, "Did not find any LoRA parameters in the encoder"
            self.unfreeze(lora_params)

            params_to_optimize = list(self.linear_classifier.parameters()) + \
                list([p for _, p in lora_params])

        elif mode == 'full_finetune':
            self.unfreeze(self.encoder) 
            self.unfreeze(self.linear_classifier)

            params_to_optimize = list(self.linear_classifier.parameters()) + \
                list(self._get_encoder_params_without_head(with_name=False))

        else:
            raise ValueError(f"Invalid mode: {mode}")

        return params_to_optimize
    
    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)
    
    def _log_metrics_cls(self, outputs, targets, prefix="train"):
        """ Calculate accuracy and other classification-specific metrics """
        acc1, acc5 = cls_metric(self.data_config, outputs, targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)

    def _log_metrics_reg(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        mse, mae = reg_metric(self.data_config, outputs[0], targets)
        self.log(f"{prefix}_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=True)

    def _get_encoder_params_without_head(self, verbose=False, with_name=True):
        if self.dot_str_of_linear_classifier is None:
            out = self.encoder.named_parameters()
        else:
            out = []
            for n,p in self.encoder.named_parameters():
                if self.dot_str_of_linear_classifier not in n:
                    out.append((n,p))
                elif verbose:
                    print(f"Skipping {n} from encoder parameters")

        if not with_name:
            out = [p for _, p in out]
        return out

    def _filter_named_params(self, named_params, targets):
        out = []
        for name, param in named_params:
            for t in targets:
                if t in name:
                    out.append((name, param))
        return out


class LightningSegmentationTask(LightningTask):

    encoder: torch.nn.Module

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.embed_dim = model_config.embed_dim
        self.criterion = nn.CrossEntropyLoss()

    def _build_default_segm_modules(self):
        edim = self.embed_dim
        data_config = self.data_config

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

    def freeze_and_return_params(self):
        segm_params = (
            list(self.neck.parameters())
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
    
    def _forward_feats(self, samples):
        """ returns features from the encoder ready to be inputted to self.neck """
        raise NotImplementedError('Subclass must implement this method')

    def forward(self, samples):
        feats = self._forward_feats(samples)
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a
    
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)