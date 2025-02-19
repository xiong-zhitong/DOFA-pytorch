from .SatMAE.models_vit_group_channels_seg import (
    vit_large_patch16 as vit_large_patch16_seg,
)
from .SatMAE.models_vit_group_channels import vit_large_patch16 as vit_large_patch16_cls

from .SatMAE.models_vit import vit_large_patch16 as vit_large_patch16_cls_rgb
from .SatMAE.models_vit_seg import vit_large_patch16 as vit_large_patch16_seg_rgb

import torch.nn as nn
import torch
import os

from .lightning_task import LightningClsRegTask, LightningSegmentationTask

from torchvision.datasets.utils import download_url

from .base import LinearHead


def load_encoder(model_config):
    URL = "https://huggingface.co/mubashir04/{}/resolve/main/{}"
    assert model_config.size == "large", "Only large size is supported for now"
    
    # get the params for the model
    kwargs = {}
    kwargs["img_size"] = model_config.image_resolution
    kwargs["patch_size"] = model_config.patch_size
    kwargs["in_chans"] = model_config.num_channels
    if model_config.num_channels == 3:
        encoder = vit_large_patch16_cls_rgb(**kwargs)
    else:
        kwargs["channel_groups"] = model_config.channel_groups
        encoder = vit_large_patch16_cls(**kwargs)

    # look for pretrained weights
    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )
        checkpoint = torch.load(model_config.pretrained_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    return encoder


class SatMAEClsReg(LightningClsRegTask):

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        self.linear_classifier = LinearHead(
            in_features=model_config.embed_dim, num_classes=data_config.num_classes
        )
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.linear_classifier(x)
        return x


class SatMAESegmentation(LightningSegmentationTask):
    url = "https://huggingface.co/mubashir04/{}/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        feats = self.encoder.forward_features(samples)
        return feats



# Model factory for different dinov2 tasks
def SatMAEModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return SatMAEClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return SatMAESegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
