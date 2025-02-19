import os
import argparse
from pathlib import Path
import torch
import pytest
import regex as re

from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer

from geofm_src.factory import model_registry
from geofm_src.datasets.data_module import BenchmarkDataModule


CONFIGS = {
    "classification": {
        "models": ["croma_cls.yaml"],
        "data_path": os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        ),
        "task": "classification",
    },
    "segmentation": {
        "models": ["croma_seg.yaml"],
        "data_path": os.path.join(
            "tests", "configs", "segmentation_dataset_config.yaml"
        ),
        "task": "segmentation",
    },
}


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

    # Save the original number of classes for later restoration
    orig_classes = data_config.num_classes

    # For mocking CROMA weights, adjust num_classes as needed (e.g. embedding size)
    data_config.num_classes = 3072

    # Instantiate the model without pretrained weights
    os.environ["MODEL_WEIGHTS_DIR"] = str(tmp_path)
    model_path = os.path.basename(model_config.pretrained_path)
    model_config.pretrained_path = None
    model_without_weights = model_class(other_args, model_config, data_config)

    # Prepare a mocked state dict.
    # In the original wrapper, weights from a sub-module (encoder) are grouped.

    mocked_path = tmp_path / model_path
    state_dict = model_without_weights.encoder.state_dict()
    groups = [
        "s1_encoder",
        "s1_GAP_FFN",
        "s2_encoder",
        "s2_GAP_FFN",
        "joint_encoder",
    ]
    new_dict = {}
    for grp in groups:
        new_dict[grp] = {}
        for key, val in state_dict.items():
            if key.startswith(grp):
                new_key = key.replace(f"{grp}.", "")
                new_dict[grp][new_key] = val
    # tODo - Add the missing weights
    # to the mocked state dict to remove raising error

    # Save the mocked weights
    torch.save(new_dict, str(mocked_path))

    # Restore the original number of classes
    data_config.num_classes = orig_classes

    # Set the pretrained_path for the model config
    model_config.pretrained_path = str(mocked_path)

    # For the classification case, expect a runtime error when loading weights
    if task == "classification":
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                'Error(s) in loading state_dict for Sequential:\n\tMissing key(s) in state_dict: "3.weight", "3.bias". '
            ),
        ):
            _ = model_class(other_args, model_config, data_config)
        # Return the model instantiated without the weights (for subsequent testing)
        return model_without_weights
    else:
        # For segmentation, load the weights normally
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
