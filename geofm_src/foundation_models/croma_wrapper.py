from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from .CROMA.use_croma import PretrainedCROMA
import torch
import os
from torchvision.datasets.utils import download_url
from .base import LinearHead


def load_encoder(model_config):
    URL = "https://huggingface.co/antofuller/CROMA/resolve/main/{}"

    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            # download the weights from HF
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )
    else:
        path = None

    modality = model_config.modality
    encoder = PretrainedCROMA(
        pretrained_path=path,
        size=model_config.size,
        modality=modality,
        image_resolution=model_config.image_resolution,)

    if modality == "optical":
        encoder.s2_GAP_FFN = torch.nn.Identity()
    elif modality == "SAR":
        encoder.s1_GAP_FFN = torch.nn.Identity()

    return encoder



class CromaClsReg(LightningClsRegTask):

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        self.linear_classifier = LinearHead(
            in_features=self.encoder.encoder_dim, num_classes=data_config.num_classes)
        self.dot_str_of_linear_classifier = None

        self.freeze_and_return_params()

    def forward(self, x):
        mod = self.model_config.modality
        x = self.encoder(**{f"{mod}_images": x})
        x = x[f"{mod}_encodings"]
        x = self.linear_classifier(x)
        return x


class CromaSegmentation(LightningSegmentationTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, x):
        mod = self.model_config.modality
        x = self.encoder(**{f'{mod}_images': x})["out_feats"]
        return x



# Model factory for different dinov2 tasks
def CromaModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return CromaClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return CromaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
