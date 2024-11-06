from foundation_models import CROMA, Panopticon, ScaleMAE, GFM, Dinov2, SoftCON, DOFA, SatMAE
from datasets.geobench_wrapper import GeoBenchDataset


model_registry = {
    "croma": CROMA,
    "panopticon": Panopticon,
    "scalemae": ScaleMAE,
    "gfm": GFM,
    "dinov2": Dinov2,
    "softcon": SoftCON,
    "dofa": DOFA,
    "satmae":SatMAE,
    # Add other model mappings here
}

dataset_registry = {
    "geobench": GeoBenchDataset,
    # Add other dataset mappings here
}


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    dataset = dataset_class(config_data)
    # return the train, val, and test dataset
    return dataset.create_dataset()


def create_model(config_model, dataset_config = None):
    model_name = config_model.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    if dataset_config is not None:
        config_model.apply_dataset(dataset_config)
    
    model = model_class(config_model)

    return model
