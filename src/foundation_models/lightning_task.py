import pytorch_lightning as pl
import torch

class LightningTask(pl.LightningModule):
    def __init__(self, args, config, data_config):
        super().__init__()
        self.config = config #model_config
        self.args = args # args for optimization params
        self.data_config = data_config # dataset_config
        self.save_hyperparameters()
    
    def loss(self, outputs, labels):
        raise NotImplementedError("This method should be implemented in task-specific classes")

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def log_metrics(self, outputs, targets, prefix="train"):
        """Abstract method for logging task-specific metrics."""
        raise NotImplementedError("This method should be implemented in task-specific classes")

    def forward(self, samples):
        raise NotImplementedError("This method should be implemented in task-specific classes")

    def training_step(self, batch, batch_idx):
        #current_lr = self.optimizers().param_groups[0]['lr']
        #print(current_lr) Debug
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="val")
        return loss
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        targets = targets.long()
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, prefix="test")
        return loss

        
    def configure_optimizers(self):
        if self.config.task == 'classification':
            optimizer = torch.optim.SGD(self.params_to_optimize(),
                           lr=self.args.lr,
                           weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.params_to_optimize(),
                           lr=self.args.lr)
        
        def lr_lambda_func(current_step: int):
            num_warmup_steps = len(self.trainer.datamodule.train_dataloader()) * self.args.warmup_epochs
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
    
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=lambda step: lr_lambda_func(self.trainer.global_step))
        
        if self.config.task == "segmentation":
            assert self.args.warmup_epochs == 3

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=2 * (self.args.epochs - self.args.warmup_epochs),
            eta_min=0.000001,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.args.warmup_epochs])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
