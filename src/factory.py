"""Factory utily functions to create datasets and models."""

from model_config import model_config_registry
from dataset_config import dataset_config_registry


def create_dataset(config_data):
    dataset_type = config_data.dataset_type
    dataset_class = dataset_config_registry.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset type '{dataset_type}' not found.")
    dataset = dataset_class(config_data)
    # return the train, val, and test dataset
    return dataset.create_dataset()


def create_model(args, config_model, dataset_config=None):
    model_name = config_model.model_type
    model_class = model_config_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    if dataset_config is not None:
        config_model.apply_dataset(dataset_config)

    model = model_class(args, config_model, dataset_config)

    return model
