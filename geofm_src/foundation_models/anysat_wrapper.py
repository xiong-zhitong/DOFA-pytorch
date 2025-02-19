import torch.nn as nn
import torch

from .lightning_task import LightningClsRegTask, LightningSegmentationTask



def load_encoder():
    encoder = torch.hub.load(
        "gastruc/anysat", "anysat", pretrained=True, flash_attn=False)
    return encoder


class AnySatClsReg(LightningClsRegTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder()

        self.linear_classifier = nn.Linear(
            model_config.embed_dim, data_config.num_classes)
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, x):
        x = {"spot": x}
        x = self.encoder(
            x, patch_size=16, output="tile", output_modality="spot")
        x = self.linear_classifier(x)
        return x


class AnySatSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder()
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        feats = self.encoder(
            samples, patch_size=16, output="dense", output_modality="aerial")
        return feats


# Model factory for different dinov2 tasks
def AnySatModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return AnySatClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return AnySatSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
