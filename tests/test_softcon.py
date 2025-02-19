import os
import argparse
from pathlib import Path
import torch
import pytest

import torch.nn as nn

from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer

from geofm_src.factory import model_registry
from geofm_src.datasets.data_module import BenchmarkDataModule

CONFIGS = {
    "classification": {
        "models": ["softcon_cls.yaml"],
        "data_path": os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        ),
        "task": "classification",
    },
    "segmentation": {
        "models": ["softcon_seg.yaml"],
        "data_path": os.path.join(
            "tests", "configs", "segmentation_dataset_config.yaml"
        ),
        "task": "segmentation",
    },
}


# have a dummy model to mock the `torch.hub.load()` during tests
class DummyDinoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Identity()

    def forward_features(self, x):
        B = x.size(0)
        return {"x_norm_patchtokens": torch.randn(B, 196, 384, device=x.device)}

    def get_intermediate_layers(self, x, layers):
        B = x.size(0)
        dummy_out = torch.randn(B, 196, 384, device=x.device)
        return [dummy_out for _ in layers]


@pytest.fixture(autouse=True)
def mock_torch_hub_load(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: DummyDinoModel())


@pytest.fixture(
    params=[
        (task, model_name) for task in CONFIGS for model_name in CONFIGS[task]["models"]
    ]
)
def task_and_config(request):
    task_type, model_config_name = request.param
    config_path = os.path.join("..", "src", "configs")
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="config", overrides=[f"model={model_config_name}"])
    return task_type, cfg.model


@pytest.fixture()
def other_args(task_and_config):
    task, _ = task_and_config
    args = argparse.Namespace()
    args.task = task
    args.lr = 0.001
    args.weight_decay = 0.0
    args.warmup_epochs = 0
    args.num_gpus = 0
    args.epochs = 1
    return args


@pytest.fixture()
def data_config(task_and_config):
    task, model_config = task_and_config
    data_config_path = CONFIGS[task]["data_path"]
    data_conf = OmegaConf.load(data_config_path)
    if "image_resolution" in model_config:
        data_conf.image_resolution = model_config.image_resolution
    if "num_channels" in model_config:
        data_conf.num_channels = model_config.num_channels
    return data_conf


@pytest.fixture()
def model(task_and_config, other_args, data_config, tmp_path: Path):
    task, model_config = task_and_config
    model_name = model_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    # Set environment variable for weights directory
    os.environ["MODEL_WEIGHTS_DIR"] = str(tmp_path)
    model_file_name = os.path.basename(model_config.pretrained_path)
    model_config.pretrained_path = None
    # Instantiate the model without pretrained weights
    model_without_weights = model_class(other_args, model_config, data_config)
    state_dict = model_without_weights.encoder.state_dict()

    # Remove weights that are expected to be randomly initialized on load
    if "linear_classifier.weight" in state_dict:
        del state_dict["linear_classifier.weight"]
    if "linear_classifier.bias" in state_dict:
        del state_dict["linear_classifier.bias"]

    mocked_path = tmp_path / model_file_name
    torch.save(state_dict, str(mocked_path))
    # Instantiate model with the "pretrained" weights
    model_config.pretrained_path = str(mocked_path)
    return model_class(other_args, model_config, data_config)


@pytest.fixture()
def datamodule(data_config):
    return BenchmarkDataModule(
        data_config, num_workers=1, batch_size=2, pin_memory=False
    )


def test_fit(model, datamodule, tmp_path: Path) -> None:
    """Run fit to test model."""
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)
    trainer.fit(model, datamodule)
