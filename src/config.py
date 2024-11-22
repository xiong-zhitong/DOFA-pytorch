from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple


# Dataset Config
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
    band_names: List[str] = []
    band_wavelengths: List[float] = []
    data_path: str = "/home/zhitong/Datasets/geobench/"
    multilabel: bool = False
    additional_param: Optional[int] = None
    ignore_index: Optional[int] = None
    image_resolution: int = 224

    @validator("benchmark_name")
    def validate_benchmark_name(cls, value):
        if value not in ["classification_v1.0", "segmentation_v1.0"]:
            raise ValueError(f"Unsupported task type: {value}")
        return value


####################################################

###### STANDARD CONFIGS ############################

####################################################


############ CLASSSIFICATION #######################

class Resics_rgb_Config(BaseDatasetConfig):
    dataset_type: str = "resisc45"
    task: str = "classification"
    dataset_name: str = "resisc45"
    num_classes: int = 45
    num_channels: int = 3
    data_path: str = "/mnt/data/datasets_classification/resisc45"
    band_names: str = ["Red", "Green", "Blue"]
    image_resolution: int = 224
    band_wavelengths: List[float] = [
        0.665,
        0.56,
        0.49
    ]
    multilabel: bool = False

class Benv2_all_Config(BaseDatasetConfig):
    dataset_type: str = "benv2"
    task: str = "classification"
    dataset_name: str = "benv2"
    num_classes: int = 19
    num_channels: int = 14
    data_path: str = "/mnt/data/datasets_classification/benv2"
    image_resolution: int = 224 # desired image size for model input
    band_wavelengths: List[float] = []
    multilabel: bool = True
    bands: str = "all" # argument for torchgeo loader

class Benv2_S1_Config(Benv2_all_Config):
    bands: str = "s1"
    num_channels: int = 2

class Benv2_S2_Config(Benv2_all_Config):
    bands: str = "s2"
    num_channels: int = 12

class Benv2_RGB_Config(Benv2_all_Config):
    bands: str = "rgb"
    num_channels: int = 3


class GeoBench_so2sat_10band_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "classification_v1.0"
    dataset_name: str = "m-so2sat"
    task: str = "classification"
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
    ]
    band_wavelengths: List[float] = [
        0.49,
        0.56,
        0.665,
        0.705,
        0.74,
        0.783,
        0.842,
        0.865,
        1.61,
        2.19,
    ]
    num_classes: int = 17
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_pv4ger_cls_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "classification_v1.0"
    dataset_name: str = "m-pv4ger"
    task: str = "classification"
    band_names: List[str] = ["Blue", "Green", "Red"]
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    num_classes: int = 2
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_brick_kiln_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "classification_v1.0"
    dataset_name: str = "m-brick-kiln"
    task: str = "classification"
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
        "12 - SWIR",
    ]
    num_classes: int = 2
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_forestnet_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "classification_v1.0"
    dataset_name: str = "m-forestnet"
    task: str = "classification"
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    num_classes: int = 12
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_eurosat_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "classification_v1.0"
    dataset_name: str = "m-eurosat"
    task: str = "classification"
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_classes: int = 10
    multilabel: bool = False
    num_channels: int = len(band_names)

###################################################

#### SCALEMAE with 3 channels #######################

###################################################

class GeoBench_brick_kiln_3_Config(GeoBench_brick_kiln_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_so2sat_3_Config(GeoBench_so2sat_10band_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_forestnet_3_Config(GeoBench_forestnet_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    num_channels: int = len(band_names)


class GeoBench_eurosat_3_Config(GeoBench_eurosat_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    num_channels: int = len(band_names)


class GeoBench_pv4ger_cls_3_Config(GeoBench_pv4ger_cls_Config):
    band_names: List[str] = [
        "Blue",
        "Green",
        "Red",
    ]
    num_channels: int = len(band_names)


####################################################

#### SOFTCON with 13 channels #######################

#####################################################
class GeoBench_brick_kiln_13_Config(GeoBench_brick_kiln_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
        "12 - SWIR",
    ]
    band_wavelengths: list[float] = [0.44, 0.49, 0.56, 0.66, 0.7, 0.74, 0.78, 0.84, 0.86, 0.94, 1.37, 1.61, 2.2]
    num_channels: int = len(band_names)


class GeoBench_brick_kiln_10_Config(GeoBench_brick_kiln_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_so2sat_13_Config(GeoBench_so2sat_10band_Config):
    band_names: List[str] = [
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_forestnet_13_Config(GeoBench_forestnet_Config):
    band_names: List[str] = [
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - NIR",
        "06 - SWIR1",
        "07 - SWIR2",
        "07 - SWIR2",
        "07 - SWIR2",
        "07 - SWIR2",
        "07 - SWIR2",
        "07 - SWIR2",
        "07 - SWIR2",
    ]
    num_channels: int = len(band_names)


class GeoBench_eurosat_13_Config(GeoBench_eurosat_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "10 - SWIR - Cirrus",
        "11 - SWIR",
        "12 - SWIR",
    ]
    band_wavelengths: list[float] = [0.44, 0.49, 0.56, 0.66, 0.7, 0.74, 0.78, 0.84, 0.86, 0.94, 1.37, 1.61, 2.2]
    num_channels: int = len(band_names)

class GeoBench_eurosat_10_Config(GeoBench_eurosat_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)


class GeoBench_pv4ger_cls_13_Config(GeoBench_pv4ger_cls_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

####################################################

########## 12 Channels CROMA - cirrus removed #######################

####################################################

class GeoBench_brick_kiln_12_Config(GeoBench_brick_kiln_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)


class GeoBench_so2sat_12_Config(GeoBench_so2sat_10band_Config):
    band_names: List[str] = [
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_forestnet_9_Config(GeoBench_forestnet_Config):
    band_names: List[str] = ['04 - Red', '03 - Green', '02 - Blue', '05 - NIR', '05 - NIR', '05 - NIR', '05 - NIR', '06 - SWIR1', '07 - SWIR2']
    band_wavelengths: List[float] = [0.66, 0.56, 0.49, 0.86, 0.86, 0.86, 0.86, 1.61, 2.2]
    num_channels: int = len(band_names)


class GeoBench_eurosat_12_Config(GeoBench_eurosat_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_pv4ger_cls_12_Config(GeoBench_pv4ger_cls_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)


############ SEGMENTATION #######################


class GeoBench_pv4ger_seg_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-pv4ger-seg"
    task: str = "segmentation"
    band_names: List[str] = ["Blue", "Green", "Red"]
    num_classes: int = 2
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_cashew_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-cashew-plant"
    task: str = "segmentation"
    band_names: List[str] = [
        "04 - Red",
        "03 - Green",
        "02 - Blue",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    band_wavelengths: List[float] = [0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2]
    num_classes: int = 7
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_chesapeake_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-chesapeake"
    task: str = "segmentation"
    band_names: List[str] = ["Blue", "Green", "Red"]
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    num_classes: int = 7
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_NeonTree_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-NeonTree"
    task: str = "segmentation"
    band_names: List[str] = ["Blue", "Green", "Red"]
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    num_classes: int = 2
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_SAcrop_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-SA-crop-type"
    task: str = "segmentation"
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    band_wavelengths: List[float] = [
        0.44,
        0.49,
        0.56,
        0.66,
        0.7,
        0.74,
        0.78,
        0.84,
        0.86,
        0.94,
        1.61,
        1.61,
        2.2,
    ]
    num_classes: int = 10
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_SAcrop_10_Config(GeoBench_SAcrop_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "12 - SWIR",
    ]
    band_wavelengths: List[float] = [
        0.44,
        0.49,
        0.56,
        0.66,
        0.7,
        0.74,
        0.78,
        0.84,
        0.86,
        2.2,
    ]
    num_channels: int = len(band_names)

class GeoBench_nzcattle_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-nz-cattle"
    task: str = "segmentation"
    band_names: List[str] = ["Red", "Green", "Blue"]
    band_wavelengths: List[float] = [0.66, 0.56, 0.48]
    num_classes: int = 2
    multilabel: bool = False
    num_channels: int = len(band_names)


class GeoBench_cashew_10band_Config(GeoBenchDatasetConfig):
    benchmark_name: str = "segmentation_v1.0"
    dataset_name: str = "m-cashew-plant"
    task: str = "segmentation"
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "11 - SWIR",
        "12 - SWIR",
    ]
    band_wavelengths: List[float] = [
        0.49,
        0.56,
        0.665,
        0.705,
        0.74,
        0.783,
        0.842,
        0.865,
        1.61,
        2.19,
    ]
    num_classes: int = 7
    multilabel: bool = False
    num_channels: int = len(band_names)



############################################################

############### SEGMENTATION 3 Channels ########################

############################################################

class GeoBench_pv4ger_seg_3_Config(GeoBench_pv4ger_seg_Config):
    band_names: List[str] = ["Blue", "Green", "Red"]
    num_channels: int = len(band_names)


class GeoBench_cashew_3_Config(GeoBench_cashew_Config):
    band_names: List[str] = [
        "04 - Red",
        "03 - Green",
        "02 - Blue",
    ]
    band_wavelengths: List[float] = [0.66, 0.56, 0.49]
    num_channels: int = len(band_names)

class GeoBench_chesapeak_3_Config(GeoBench_chesapeake_Config):
    band_names: List[str] = ["Blue", "Green", "Red"]
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    num_channels: int = len(band_names)

class GeoBench_NeonTree_3_Config(GeoBench_NeonTree_Config):
    band_names: List[str] = ["Blue", "Green", "Red"]
    band_wavelengths: List[float] = [0.48, 0.56, 0.66]
    num_channels: int = len(band_names)

class GeoBench_SAcrop_3_Config(GeoBench_SAcrop_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
    ]
    band_wavelengths: List[float] = [0.49, 0.56, 0.665]
    num_channels: int = len(band_names)

class GeoBench_nzcattle_3_Config(GeoBench_nzcattle_Config):
    band_names: List[str] = ["Red", "Green", "Blue"]
    band_wavelengths: List[float] = [0.66, 0.56, 0.48]
    num_channels: int = len(band_names)


###############################################################

############## SEGMENTATION 12 Channels ########################

###############################################################


class GeoBench_pv4ger_seg_12_Config(GeoBench_pv4ger_seg_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_cashew_12_Config(GeoBench_cashew_Config):
    band_names: List[str] = [
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "11 - SWIR",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_cashew_10_Config(GeoBench_cashew_Config):
    band_names: List[str] = [
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_chesapeak_12_Config(GeoBench_chesapeake_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_NeonTree_12_Config(GeoBench_NeonTree_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_SAcrop_12_Config(GeoBench_SAcrop_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "09 - Water vapour",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_nzcattle_12_Config(GeoBench_nzcattle_Config):
    band_names: List[str] = [
        "Red",
        "Green",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
        "Blue",
    ]
    num_channels: int = len(band_names)

##############################################################

################## SEGMENTATION 13 channels ###################

##############################################################

class GeoBench_pv4ger_seg_13_Config(GeoBench_pv4ger_seg_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_cashew_13_Config(GeoBench_cashew_Config):
    band_names: List[str] = [
        "02 - Blue",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "11 - SWIR",
        "11 - SWIR",
        "11 - SWIR",
        "12 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_chesapeak_13_Config(GeoBench_chesapeake_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_NeonTree_13_Config(GeoBench_NeonTree_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)

class GeoBench_SAcrop_13_Config(GeoBench_SAcrop_Config):
    band_names: List[str] = [
        "01 - Coastal aerosol",
        "02 - Blue",
        "03 - Green",
        "04 - Red",
        "05 - Vegetation Red Edge",
        "06 - Vegetation Red Edge",
        "07 - Vegetation Red Edge",
        "08 - NIR",
        "08A - Vegetation Red Edge",
        "09 - Water vapour",
        "09 - Water vapour",
        "12 - SWIR",
        "12 - SWIR",
    ]
    num_channels: int = len(band_names)

class GeoBench_nzcattle_13_Config(GeoBench_nzcattle_Config):
    band_names: List[str] = [
        "Blue",
        "Blue",
        "Green", 
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
        "Red",
    ]
    num_channels: int = len(band_names)



dataset_config_registry = {
    ##### classification #####
    "geobench_so2sat_cls": GeoBench_so2sat_10band_Config,
    "geobench_pv4ger_cls": GeoBench_pv4ger_cls_Config,
    "geobench_so2sat_10band": GeoBench_so2sat_10band_Config,
    "geobench_forestnet_13": GeoBench_forestnet_13_Config,
    "geobench_so2sat_13": GeoBench_so2sat_13_Config,
    "geobench_brick_kiln_13": GeoBench_brick_kiln_13_Config,
    "geobench_brick_kiln_10": GeoBench_brick_kiln_10_Config,
    "geobench_eurosat_13": GeoBench_eurosat_13_Config,
    "geobench_eurosat_10": GeoBench_eurosat_10_Config,
    "geobench_forestnet_3": GeoBench_forestnet_3_Config,
    "geobench_so2sat_3": GeoBench_so2sat_3_Config,
    "geobench_brick_kiln_3": GeoBench_brick_kiln_3_Config,
    "geobench_eurosat_3": GeoBench_eurosat_3_Config,
    "geobench_brick_kiln_12": GeoBench_brick_kiln_12_Config,
    "geobench_so2sat_12": GeoBench_so2sat_12_Config,
    "geobench_forestnet_9": GeoBench_forestnet_9_Config,
    "geobench_eurosat_12": GeoBench_eurosat_12_Config,
    "geobench_pv4ger_cls_12": GeoBench_pv4ger_cls_12_Config,
    ######## segmentation ##########
    "geobench_pv4ger_seg": GeoBench_pv4ger_seg_Config,
    "geobench_cashew": GeoBench_cashew_Config,
    "geobench_cashew_10": GeoBench_cashew_10band_Config,
    "geobench_chesapeake": GeoBench_chesapeake_Config,
    "geobench_NeonTree": GeoBench_NeonTree_Config,
    "geobench_nzcattle": GeoBench_nzcattle_Config,
    "geobench_SAcrop": GeoBench_SAcrop_Config,
    "geobench_SAcrop_10": GeoBench_SAcrop_10_Config,
    "geobench_pv4ger_seg_3": GeoBench_pv4ger_seg_3_Config,
    "geobench_cashew_3": GeoBench_cashew_3_Config,
    "geobench_chesapeak_3": GeoBench_chesapeak_3_Config,
    "geobench_NeonTree_3": GeoBench_NeonTree_3_Config,
    "geobench_SAcrop_3": GeoBench_SAcrop_3_Config,
    "geobench_nzcattle_3": GeoBench_nzcattle_3_Config,
    "geobench_pv4ger_seg_12": GeoBench_pv4ger_seg_12_Config,
    "geobench_cashew_12": GeoBench_cashew_12_Config,
    "geobench_chesapeak_12": GeoBench_chesapeak_12_Config,
    "geobench_NeonTree_12": GeoBench_NeonTree_12_Config,
    "geobench_SAcrop_12": GeoBench_SAcrop_12_Config,
    "geobench_nzcattle_12": GeoBench_nzcattle_12_Config,
    "geobench_pv4ger_seg_13": GeoBench_pv4ger_seg_13_Config,
    "geobench_cashew_13": GeoBench_cashew_13_Config,
    "geobench_chesapeak_13": GeoBench_chesapeak_13_Config,
    "geobench_NeonTree_13": GeoBench_NeonTree_13_Config,
    "geobench_SAcrop_13": GeoBench_SAcrop_13_Config,
    "geobench_nzcattle_13": GeoBench_nzcattle_13_Config,
    ####### other datasets ######
    "resics45_rgb": Resics_rgb_Config,
    "benv2_s1": Benv2_S1_Config,
    "benv2_s2": Benv2_S2_Config,
    "benv2_rgb": Benv2_RGB_Config,
    "benv2_all": Benv2_all_Config
}

#######################################################################################


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
        cls.band_wavelengths = dataset_config.band_wavelengths
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
    pretrained_path: str = (
        "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_sentinel.pth"
    )

class SatMAE_seg_rgb_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "segmentation"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    num_channels: int = 3
    image_resolution: int = 224
    patch_size: int = 16
    pretrained_path: str = (
        "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth"
    )


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
    pretrained_path: str = (
        "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_sentinel.pth"
    )

class SatMAE_cls_rgb_Config(BaseModelConfig):
    model_type: str = "satmae"
    out_features: bool = True
    task: str = "classification"
    freeze_backbone: bool = True
    embed_dim: int = 1024
    num_channels: int = 3
    patch_size: int = 16
    image_resolution: int = 224
    pretrained_path: str = (
        "src/fm_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth"
    )


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
    dino_size: str = "dinov2_vitl14"
    image_resolution: int = 224
    out_features: bool = True
    freeze_backbone: bool = True
    task: str = "segmentation"
    embed_dim: int = 1024
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
        if value not in ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']:
            raise ValueError("Wrong Dinov2 model.")
        return value

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
