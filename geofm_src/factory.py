"""Factory utily functions to create datasets and models."""

from geofm_src.foundation_models import (
    CromaModel,
    ScaleMAEModel,
    GFMModel,
    DinoV2Model,
    SoftConModel,
    DofaModel,
    SatMAEModel,
    AnySatModel,
    SenPaMAEModel,
    StealthModel
)
from geofm_src.datasets.geobench_wrapper import GeoBenchDataset
from geofm_src.datasets.resisc_wrapper import Resics45Dataset
from geofm_src.datasets.benv2_wrapper import BenV2Dataset
from geofm_src.datasets.spectral_earth_wrapper import CorineDataset
from geofm_src.datasets.digital_typhoon_wrapper import DigitalTyphoonDataset
from geofm_src.datasets.tropical_cyclone_wrapper import TropicalCycloneDataset
from geofm_src.datasets.hyperview_wrapper import HyperviewDataset
from geofm_src.datasets.dummy_dataset import DummyWrapper

model_registry = {
    "croma": CromaModel,
    "scalemae": ScaleMAEModel,
    "gfm": GFMModel,
    "dinov2": DinoV2Model,
    "softcon": SoftConModel,
    "dofa": DofaModel,
    "satmae": SatMAEModel,
    "anysat": AnySatModel,
    "senpamae": SenPaMAEModel,
    "stealth": StealthModel,
    # Add other model mappings here
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    "resisc45": Resics45Dataset,
    "benv2": BenV2Dataset,
    "corine": CorineDataset,
    "digital_typhoon": DigitalTyphoonDataset,
    "tropical_cyclone": TropicalCycloneDataset,
    "hyperview": HyperviewDataset,
    # Add other dataset mappings here
    "dummy": DummyWrapper,
    # Add other dataset mappings here
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    
    dataset = dataset_class(config_data)

    return dataset.create_dataset()


def create_model(args, model_config, dataset_config=None):
    model_name = model_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        if model_name == 'stealth':
            raise ImportError("Stealth model requires the stealth_wrapper to be available")
        raise ValueError(f"Model type '{model_name}' not found.")

    model = model_class(args, model_config, dataset_config)

    return model
