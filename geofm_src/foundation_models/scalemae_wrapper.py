from .ScaleMAE.models_vit_seg import vit_large_patch16 as vit_large_patch16_seg
from .ScaleMAE.models_vit import vit_large_patch16 as vit_large_patch16_cls

import torch.nn as nn
import torch
import os

# use mmsegmentation for upernet+mae
from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from torchvision.datasets.utils import download_url

from .base import LinearHead


def load_encoder(model_config):
    URL = "https://huggingface.co/torchgeo/{}/resolve/main/{}"
    assert model_config.size == 'large', "Only large size is supported for now"

    encoder = vit_large_patch16_cls(
        img_size=model_config.image_resolution, in_chans=model_config.num_channels, drop_path_rate=0.0)

    # Load pretrained weights
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


class ScaleMAEClsReg(LightningClsRegTask):

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


class ScaleMAESegmentation(LightningSegmentationTask):
    url = "https://huggingface.co/torchgeo/{}/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        
        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, x):
        return self.encoder(x)



# Model factory for different dinov2 tasks
def ScaleMAEModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return ScaleMAEClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return ScaleMAESegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
