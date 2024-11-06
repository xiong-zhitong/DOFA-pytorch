from pydantic import BaseModel, Field, validator
from typing import Optional, Dict

#Dataset Config
class BaseDatasetConfig(BaseModel):
    dataset_type: str
    task: str  # e.g., "classification", "segmentation"
    num_classes: int
    num_channels: int
    data_path: str

    @validator("task")
    def validate_task(cls, value):
        if value not in ["classification", "segmentation"]:
            raise ValueError(f"Unsupported task type: {value}")
        return value

# Example dataset configurations
class GeoBenchDatasetConfig(BaseDatasetConfig):
    dataset_type: str = "geobench"
    benchmark_name: str = ""
    dataset_name: str = ""
    band_names: list = []
    band_wavelengths: list = []
    data_path: str = "/home/zhitong/Datasets/geobench/"
    multilabel: bool = False
    additional_param: Optional[int] = None
    ignore_index: int = None
    image_resolution = 224

    @validator("benchmark_name")
    def validate_benchmark_name(cls, value):
        if value not in ["classification_v1.0", "segmentation_v1.0"]:
            raise ValueError(f"Unsupported task type: {value}")
        return value


class GeoBench_so2sat_Config(GeoBenchDatasetConfig):
    benchmark_name = "classification_v1.0"
    dataset_name = "m-so2sat"
    task = "classification"
    band_names = ['02 - Blue', '02 - Blue', '03 - Green', '04 - Red', \
            '05 - Vegetation Red Edge', '06 - Vegetation Red Edge',   \
            '07 - Vegetation Red Edge', '08 - NIR', '08A - Vegetation Red Edge',\
            '11 - SWIR', '11 - SWIR', '12 - SWIR']
    num_classes = 17
    multilabel = False
    num_channels = len(band_names)


class GeoBench_pv4ger_cls_Config(GeoBenchDatasetConfig):
    benchmark_name = "classification_v1.0"
    dataset_name = "m-pv4ger"
    task = "classification"
    band_names = ['Blue', 'Green', 'Red']
    band_wavelengths = [0.48, 0.56, 0.66]
    num_classes = 2
    multilabel = False
    num_channels = len(band_names)

class GeoBench_pv4ger_seg_Config(GeoBenchDatasetConfig):
    benchmark_name = "segmentation_v1.0"
    dataset_name = "m-pv4ger-seg"
    task = "segmentation"
    band_names = ['Blue', 'Green', 'Red']
    num_classes = 2
    band_wavelengths = [0.48, 0.56, 0.66]
    multilabel = False
    num_channels = len(band_names)


class GeoBench_cashew_Config(GeoBenchDatasetConfig):
    benchmark_name = "segmentation_v1.0"
    dataset_name = "m-cashew-plant"
    task = "segmentation"
    band_names = ['01 - Coastal aerosol', '02 - Blue', '03 - Green', '04 - Red',\
            '05 - Vegetation Red Edge', '06 - Vegetation Red Edge', '07 - Vegetation Red Edge',\
            '08 - NIR', '08A - Vegetation Red Edge', '09 - Water vapour', '11 - SWIR', '11 - SWIR', '12 - SWIR']
    num_classes = 7
    band_wavelengths = [0.44, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 0.945, 1.61, 1.61, 2.19]
    multilabel = False
    num_channels = len(band_names)


class GeoBench_cashew_10band_Config(GeoBenchDatasetConfig):
    benchmark_name = "segmentation_v1.0"
    dataset_name = "m-cashew-plant"
    task = "segmentation"
    band_names = ['02 - Blue', '03 - Green', '04 - Red',\
            '05 - Vegetation Red Edge', '06 - Vegetation Red Edge', '07 - Vegetation Red Edge',\
            '08 - NIR', '08A - Vegetation Red Edge', '11 - SWIR', '12 - SWIR']
    num_classes = 7
    band_wavelengths = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
    multilabel = False
    num_channels = len(band_names)


dataset_config_registry = {
    "geobench_so2sat" : GeoBench_so2sat_Config,
    "geobench_pv4ger_seg" : GeoBench_pv4ger_seg_Config,
    "geobench_pv4ger_cls": GeoBench_pv4ger_cls_Config,
    "geobench_cashew" : GeoBench_cashew_Config,
    "geobench_cashew_10band": GeoBench_cashew_10band_Config,
}


#######################################################################################

# Model Config
class BaseModelConfig(BaseModel):
    model_type: str
    num_classes: int = 10
    task: Optional[str] = "classification"
    band_wavelengths: Optional[list] = None
    freeze_backbone: bool = True
    image_resolution: int = 224
    additional_params: Optional[Dict] = Field(default_factory=dict)
    num_channels: int = Field(12)
    
    def apply_dataset(cls, dataset_config: BaseDatasetConfig):
        cls.num_classes = dataset_config.num_classes
        cls.task = dataset_config.task
        cls.num_channels = dataset_config.num_channels
        # image_resolution depends on the model
        dataset_config.image_resolution = cls.image_resolution
        # model wavelength determined by dataset
        cls.band_wavelengths = dataset_config.band_wavelengths

    @validator("model_type")
    def validate_model_type(cls, value):
        if value not in ["scale-mae", "croma", "dofa"]:
            raise ValueError(f"Unsupported model type: {value}")
        return value

    class Config:
        validate_assignment = True


class SatMAE_seg_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features = True
    task = 'segmentation'
    freeze_backbone = True
    embed_dim = 1024
    image_resolution: int = 96
    patch_size: int = 8
    num_channels: int = 10
    channel_groups: int = ((0, 1, 2, 6), (3, 4, 5, 7), (8, 9))
    pretrained_path: str = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/checkpoint_ViT-L_pretrain_fmow_sentinel.pth"


class Panopticon_seg_Config(BaseModelConfig):
    model_type: str = "panopticon"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/panopticon-v1-1103"
    image_resolution = 224
    out_features = True
    task = 'segmentation'
    freeze_backbone = True
    embed_dim = 768
    ds_name = "ben-s2"
    full_spectra = False



class CROMA_cls_Config(BaseModelConfig):
    model_type: str = "croma"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/CROMA_base.pt"
    size = 'base'
    modality = 'optical'
    image_resolution = 120
    out_features = True
    freeze_backbone = True
    task = 'classification'
    num_channels: int = 12  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 12:
            raise ValueError("CROMA requires #channels to be 12!")
        return value

    class Config:
        validate_assignment = True


class CROMA_seg_Config(BaseModelConfig):
    model_type: str = "croma"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/CROMA_base.pt"
    size = 'base'
    modality = 'optical'
    image_resolution = 120
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 768
    num_channels: int = 12  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 12:
            raise ValueError("CROMA requires #channels to be 12!")
        return value

    class Config:
        validate_assignment = True



class ScaleMAE_seg_Config(BaseModelConfig):
    model_type: str = "scalemae"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/scalemae-vitlarge-800.pth"
    image_resolution = 224
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 1024
    num_channels: int = 3  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("ScaleMAE requires #channels to be 3!")
        return value

    class Config:
        validate_assignment = True



class Dinov2_seg_Config(BaseModelConfig):
    model_type: str = "dinov2"
    dino_size = "dinov2_vitl14"
    image_resolution = 224
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 1024
    num_channels: int = 3  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("ScaleMAE requires #channels to be 3!")
        return value

    class Config:
        validate_assignment = True


class Dinov2_cls_Config(BaseModelConfig):
    model_type: str = "dinov2"
    dino_size = "dinov2_vitl14"
    image_resolution = 224
    out_features = True
    freeze_backbone = True
    task = 'classification'
    embed_dim = 1024
    num_channels: int = 3  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("ScaleMAE requires #channels to be 3!")
        return value

class Dinov2base_seg_Config(Dinov2_seg_Config):
    model_type: str = "dinov2"
    dino_size = "dinov2_vitb14"
    embed_dim = 768



class Dinov2basereg_seg_Config(Dinov2_seg_Config):
    model_type: str = "dinov2"
    dino_size = "dinov2_vitb14_reg"
    embed_dim = 768




class SoftCON_seg_Config(BaseModelConfig):
    model_type: str = "softcon"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/B13_vitb14_softcon.pth"
    image_resolution = 224
    softcon_size = "vit_base"
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 768
    num_channels: int = 13  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 13:
            raise ValueError("SoftCON requires #channels to be 13!")
        return value

    class Config:
        validate_assignment = True



class DOFA_seg_Config(BaseModelConfig):
    model_type: str = "dofa"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/DOFA-pytorch/fm_weights/DOFA_ViT_large_e100.pth"
    image_resolution = 224
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 1024
    dofa_size = 'dofa_large'

    @validator("dofa_size")
    def validate_num_channels(cls, value):
        if value not in ['dofa_base', 'dofa_large']:
            raise ValueError("DOFA model size should be dofa_base or dofa_large !")
        return value

    class Config:
        validate_assignment = True



class GFM_seg_Config(BaseModelConfig):
    model_type: str = "gfm"
    pretrained_path = "/home/zhitong/OFALL/OFALL_baseline/mae/eval-fm/fm_weights/gfm.pth"
    image_resolution = 192
    out_features = True
    freeze_backbone = True
    task = 'segmentation'
    embed_dim = 1024
    num_channels: int = 3  # Define the field for num_channels

    @validator("num_channels")
    def validate_num_channels(cls, value):
        if value != 3:
            raise ValueError("GFM requires #channels to be 3!")
        return value

    class Config:
        validate_assignment = True

#
model_config_registry = {
    "croma_cls": CROMA_cls_Config,
    "croma_seg": CROMA_seg_Config,
    "panopticon_seg": Panopticon_seg_Config,
    "scalemae_seg": ScaleMAE_seg_Config,
    "gfm_seg": GFM_seg_Config,
    "dinov2_seg": Dinov2_seg_Config,
    "dinov2base_seg": Dinov2base_seg_Config,
    "softcon_seg": SoftCON_seg_Config,
    "dofa_seg": DOFA_seg_Config,
    "satmae_seg": SatMAE_seg_Config,
    "dinov2_cls": Dinov2_cls_Config,
}
