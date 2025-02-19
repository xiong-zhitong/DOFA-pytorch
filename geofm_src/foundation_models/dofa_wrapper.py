import torch
import os
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls
from .DOFA.models_dwv_seg import vit_base_patch16 as vit_base_patch16_seg
from .DOFA.models_dwv_seg import vit_large_patch16 as vit_large_patch16_seg
from .lightning_task import LightningClsRegTask, LightningSegmentationTask
from timm.models.layers import trunc_normal_
from torchvision.datasets.utils import download_url
from peft import LoraConfig, get_peft_model

from .base import LinearHead



def load_encoder(model_config):
    URL = "https://huggingface.co/earthflow/dofa/resolve/main/{}"

    encoder = (
        vit_large_patch16_cls()
        if model_config.size == "large"
        else vit_base_patch16_cls())

    if model_config.get("pretrained_path", None):
        path = model_config.pretrained_path
        if not os.path.exists(path):
            # download the weights from HF
            download_url(
                URL.format(os.path.basename(path)),
                os.path.dirname(path),
                filename=os.path.basename(path),
            )

        # Load pretrained weights
        check_point = torch.load(path, map_location="cpu")
        encoder.load_state_dict(check_point, strict=False)

    return encoder



class DinoV2ClsReg(LightningClsRegTask):

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)

        if self.training_mode == 'lora':
            self.apply_peft(self.encoder, lora_cfg=model_config.lora)

        # load linear classifier
        trunc_normal_(self.encoder.head.weight, std=0.01)
        self.linear_classifier = LinearHead(
            self.encoder.head.in_features, data_config.num_classes
        )
        self.encoder.head = self.linear_classifier
        self.dot_str_of_linear_classifier = 'head'

        self.freeze_and_return_params()


    def apply_peft(self, encoder, lora_cfg: dict):
        """
        Apply LoRA to the last few layers of the encoder using PEFT.
        """

        print("LORA: Applying PEFT: ", lora_cfg)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_rank", 16),  # Rank of LoRA
            lora_alpha=lora_cfg.get("lora_alpha", 16),  # Scaling factor for LoRA
            target_modules=lora_cfg.get(
                "lora_target_modules", "blocks.*.attn.qkv"
            ),  # ["qkv", "proj"]
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),  # Dropout rate for LoRA
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get(
                "lora_task_type", None
            ),  # Task type (use appropriate type for your model), "SEQ_CLS"
        )

        # Wrap the encoder with PEFT
        self.encoder = get_peft_model(encoder, peft_config)

    def forward(self, x):
        x, _ = self.encoder(x, self.data_config.band_wavelengths)
        return x



class DofaSegmentation(LightningSegmentationTask):
    url = "https://huggingface.co/earthflow/dofa/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.encoder = load_encoder(model_config)
        self._build_default_segm_modules()
        self.freeze_and_return_params()

    def _forward_feats(self, samples):
        feats = self.encoder(samples, self.data_config.band_wavelengths)
        return feats


# Model factory for different dinov2 tasks
def DofaModel(args, model_config, data_config):
    task = data_config.task
    if task in ["classification",'regression']:
        return DinoV2ClsReg(args, model_config, data_config)
    elif task == "segmentation":
        return DofaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
