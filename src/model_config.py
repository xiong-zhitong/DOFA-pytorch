from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple
from dataset_config import BaseDatasetConfig


# Model Config
class BaseModelConfig(BaseModel):
    model_type: str
    num_classes: int = 10
    task: Optional[str] = "classification"
    band_wavelengths: Optional[List[float]] = None
    freeze_backbone: bool = True
    image_resolution: int = 224
    additional_params: Optional[Dict] = Field(default_factory=dict)
    num_channels: int = 12
    multilabel: bool = False

    def apply_dataset(cls, dataset_config: BaseDatasetConfig):
        cls.num_classes = dataset_config.num_classes
        cls.task = dataset_config.task
        cls.num_channels = dataset_config.num_channels
        # image_resolution depends on the model
        dataset_config.image_resolution = cls.image_resolution
        # model wavelength determined by dataset
        if len(dataset_config.band_wavelengths) == 0:
            dataset_config.set_band_wavelengths()
            if dataset_config.band_wavelengths[0] is None:
                raise ValueError("Unknown band name")
        # cls.band_wavelengths = dataset_config.band_wavelengths
        cls.multilabel = dataset_config.multilabel


class SatMAE_seg_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "segmentation"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    image_resolution: int = 96
    patch_size: int = 8
    num_channels: int = 10
    channel_groups: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
    pretrained_path: str = "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_sentinel.pth"


class SatMAE_seg_rgb_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "segmentation"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    num_channels: int = 3
    image_resolution: int = 224
    patch_size: int = 16
    pretrained_path: str = "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth"


class SatMAE_cls_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "classification"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    image_resolution: int = 96
    patch_size: int = 8
    num_channels: int = 10
    channel_groups: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
    pretrained_path: str = "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_sentinel.pth"


class SatMAE_cls_rgb_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "classification"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    num_channels: int = 3
    patch_size: int = 16
    image_resolution: int = 224
    pretrained_path: str = "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth"


class Panopticon_seg_Config(BaseModelConfig):
    model_type: str = "panopticon"
    pretrained_path: str = "src/fm_weights/v2_11-06"
    image_resolution: int = 224
    out_features: bool = True
    task: str = "segmentation"
    freeze_backbone: bool = True
    embed_dim: int = 768
    ds_name: str = "ben-s2"
    full_spectra: bool = False


class Panopticon_cls_Config(BaseModelConfig):
    model_type: str = "panopticon"
    pretrained_path: str = "src/fm_weights/v2_11-06"
    image_resolution: int = 224
    out_features: bool = True
    task: str = "classification"
    freeze_backbone: bool = True
    embed_dim: int = 768
    ds_name: str = "ben-s2"
    full_spectra: bool = False


class CROMA_cls_Config(BaseModelConfig):
    model_type: str = "croma"
    pretrained_path: str = "src/fm_weights/CROMA_base.pt"
    size: str = "base"
    modality: str = "optical"
    image_resolution: int = 120
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    num_channels: int = 12

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 12:
            raise ValueError("CROMA requires #channels to be 12!")
        return value

    class Config:
        validate_assignment = True


class CROMA_seg_Config(BaseModelConfig):
    model_type: str = "croma"
    pretrained_path: str = "src/fm_weights/CROMA_base.pt"
    size: str = "base"
    modality: str = "optical"
    image_resolution: int = 120
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 768
    num_channels: int = 12

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 12:
            raise ValueError("CROMA requires #channels to be 12!")
        return value

    class Config:
        validate_assignment = True


class ScaleMAE_seg_Config(BaseModelConfig):
    model_type: str = "scalemae"
    pretrained_path: str = "src/fm_weights/scalemae-vitlarge-800.pth"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 1024
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("ScaleMAE requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class ScaleMAE_cls_Config(BaseModelConfig):
    model_type: str = "scalemae"
    pretrained_path: str = "src/fm_weights/scalemae-vitlarge-800.pth"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 1024
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("ScaleMAE requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class Dinov2_seg_Config(BaseModelConfig):
    model_type: str = "dinov2"
    dino_size: str = "dinov2_vitb14"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 768
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("Dinov2 requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class Dinov2_cls_Config(BaseModelConfig):
    model_type: str = "dinov2"
    dino_size: str = "dinov2_vitl14"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 1024
    num_channels: int = 3

    @validator("dino_size")
    def validate_dino_size(cls, value):
        if value not in [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]:
            raise ValueError("Wrong Dinov2 model.")
        return value

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("Dinov2 requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class AnySat_cls_Config(BaseModelConfig):
    model_type: str = "anysat"
    dino_size: str = "dinov2_vitl14"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 768
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("Dinov2 requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class Dinov2_base_cls_Config(Dinov2_cls_Config):
    dino_size: str = "dinov2_vitb14"
    embed_dim: int = 768


class Dinov2_base_seg_Config(Dinov2_seg_Config):
    dino_size: str = "dinov2_vitb14"
    embed_dim: int = 768


class SoftCON_seg_Config(BaseModelConfig):
    model_type: str = "softcon"
    pretrained_path: str = "src/fm_weights/B13_vitb14_softcon.pth"
    image_resolution: int = 224
    softcon_size: str = "vit_base"
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 768
    num_channels: int = 13

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 13:
            raise ValueError("SoftCON requires num_channels to be 13.")
        return value

    class Config:
        validate_assignment = True


class SoftCON_cls_Config(BaseModelConfig):
    model_type: str = "softcon"
    pretrained_path: str = "src/fm_weights/B13_vitb14_softcon.pth"
    image_resolution: int = 224
    softcon_size: str = "vit_base"
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 768
    num_channels: int = 13

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 13:
            raise ValueError("SoftCON requires num_channels to be 13.")
        return value

    class Config:
        validate_assignment = True


class DOFA_seg_Config(BaseModelConfig):
    model_type: str = "dofa"
    pretrained_path: str = "src/fm_weights/DOFA_ViT_large_e100.pth"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 1024
    dofa_size: str = "dofa_large"

    @validator("dofa_size")
    def validate_dofa_size(cls, value):
        if value not in ["dofa_base", "dofa_large"]:
            raise ValueError("DOFA model size should be 'dofa_base' or 'dofa_large'.")
        return value

    class Config:
        validate_assignment = True


class DOFA_base_seg_Config(BaseModelConfig):
    model_type: str = "dofa"
    pretrained_path: str = "src/fm_weights/DOFA_ViT_base_e120.pth"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 768
    dofa_size: str = "dofa_base"

    @validator("dofa_size")
    def validate_dofa_size(cls, value):
        if value not in ["dofa_base", "dofa_large"]:
            raise ValueError("DOFA model size should be 'dofa_base' or 'dofa_large'.")
        return value

    class Config:
        validate_assignment = True


class DOFA_cls_Config(BaseModelConfig):
    model_type: str = "dofa"
    pretrained_path: str = "src/fm_weights/DOFA_ViT_large_e100.pth"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 1024
    dofa_size: str = "dofa_large"

    @validator("dofa_size")
    def validate_dofa_size(cls, value):
        if value not in ["dofa_base", "dofa_large"]:
            raise ValueError("DOFA model size should be 'dofa_base' or 'dofa_large'.")
        return value

    class Config:
        validate_assignment = True


class GFM_seg_Config(BaseModelConfig):
    model_type: str = "gfm"
    pretrained_path: str = "src/fm_weights/gfm.pth"
    image_resolution: int = 192
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 1024
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("GFM requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


class GFM_cls_Config(BaseModelConfig):
    model_type: str = "gfm"
    pretrained_path: str = "src/fm_weights/gfm.pth"
    image_resolution: int = 192
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "classification"
    embed_dim: int = 1024
    num_channels: int = 3

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("GFM requires num_channels to be 3.")
        return value

    class Config:
        validate_assignment = True


model_config_registry = {
    "croma_cls": CROMA_cls_Config,
    "croma_seg": CROMA_seg_Config,
    "panopticon_seg": Panopticon_seg_Config,
    "panopticon_cls": Panopticon_cls_Config,
    "scalemae_seg": ScaleMAE_seg_Config,
    "scalemae_cls": ScaleMAE_cls_Config,
    "gfm_seg": GFM_seg_Config,
    "gfm_cls": GFM_cls_Config,
    "dinov2_seg": Dinov2_seg_Config,
    "dinov2_cls": Dinov2_cls_Config,
    "anysat_cls": AnySat_cls_Config,
    "dinov2_base_seg": Dinov2_base_seg_Config,
    "dinov2_base_cls": Dinov2_base_cls_Config,
    "softcon_seg": SoftCON_seg_Config,
    "softcon_cls": SoftCON_cls_Config,
    "dofa_seg": DOFA_seg_Config,
    "dofa_base_seg": DOFA_base_seg_Config,
    "dofa_cls": DOFA_cls_Config,
    "satmae_seg": SatMAE_seg_Config,
    "satmae_cls": SatMAE_cls_Config,
    "satmae_cls_rgb": SatMAE_cls_rgb_Config,
    "satmae_seg_rgb": SatMAE_seg_rgb_Config,
}
