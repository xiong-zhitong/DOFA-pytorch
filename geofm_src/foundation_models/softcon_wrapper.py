import torch.nn as nn
import torch
import os

# use mmsegmentation for upernet+mae
from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from einops import rearrange
from torchvision.datasets.utils import download_url

from .base import LinearHead


def load_encoder(model_config): 
    URL = "https://huggingface.co/wangyi111/softcon/resolve/main/{}"

    # load dino model & apply softcon changes
    encoder = torch.hub.load("facebookresearch/dinov2", model_config.dinov2_torchhub_id)
    embed_dim = encoder.num_features
    num_patches = 1 + int((224 / 14) **2)
    encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    encoder.patch_embed.proj = torch.nn.Conv2d(
        model_config.num_channels, embed_dim, kernel_size=(14, 14), stride=(14, 14)
    )


    # look for Softcon pretrained weights
    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )

        ckpt = torch.load(path, map_location="cpu")
        msg = encoder.load_state_dict(ckpt)
        print(msg)

    return encoder


class SoftConClsReg(LightningClsRegTask):
    """SoftCon model for classification."""


    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        # TODO this embed dim could be pulled from the encoder, to remove the need for the arg
        self.linear_classifier = LinearHead(
            in_features=self.encoder.num_features, num_classes=data_config.num_classes
        )
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = x["x_norm_patchtokens"].mean(dim=1)
        x = self.linear_classifier(x)
        return x


class SoftConSegmentation(LightningSegmentationTask):
    """SoftCon Model for Segmentation."""

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        outputs = self.encoder.get_intermediate_layers(samples, [4, 6, 10, 11])
        feats = [
            rearrange(out, "n (h w) c -> n c h w", h=int(out.size(1) ** 0.5))
            for out in outputs
        ]
        return feats




# Model factory for different dinov2 tasks
def SoftConModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return SoftConClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return SoftConSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
