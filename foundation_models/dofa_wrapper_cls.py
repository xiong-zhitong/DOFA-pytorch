import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_f1_score
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls
from loguru import logger
from util.misc import resize

class DOFAClassification(pl.LightningModule):
    def __init__(self, config):
        super(DOFAClassification, self).__init__()
        self.config = config
        self.encoder = vit_base_patch16_cls(num_classes=config.num_classes) if self.config.dofa_size == 'dofa_base' else vit_large_patch16_cls(num_classes=config.num_classes)
        check_point = torch.load(config.pretrained_path)
        self.encoder.load_state_dict(check_point, strict=False)
        self.out_features = config.out_features
        if config.freeze_backbone:
            self.freeze(self.encoder)
        trunc_normal_(self.encoder.head.weight, std=0.01)
        self.encoder.head = torch.nn.Sequential(torch.nn.BatchNorm1d(self.encoder.head.in_features, affine=False, eps=1e-6), self.encoder.head)
        self.unfreeze(self.encoder.head)

    def params_to_optimize(self):
        return self.encoder.head.parameters()

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def forward(self, samples):
        out_logits, feats = self.encoder(samples, self.config.band_wavelengths)
        if self.out_features:
            return out_logits, feats
        else:
            return out_logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        val_loss = self.criterion(outputs, targets)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        acc1 = self.calculate_accuracy(outputs, targets)
        self.log('val_acc1', acc1, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_loss = self.criterion(outputs, targets)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True, logger=True)
        acc1 = self.calculate_accuracy(outputs, targets)
        self.log('test_acc1', acc1, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.params_to_optimize(), lr=self.config.lr)
        return optimizer

    def calculate_accuracy(self, outputs, targets):
        _, preds = torch.max(outputs, dim=1)
        acc = torch.sum(preds == targets).float() / len(targets)
        return acc
