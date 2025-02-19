import os
import argparse
from pathlib import Path
import torch
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from lightning import Trainer

from geofm_src.factory import model_registry
from geofm_src.datasets.data_module import BenchmarkDataModule


CONFIGS = {
    "classification": {
        "models": ["scalemae_cls.yaml", "scalemae_cls_selective_params.yaml"],
        "data_path": os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        ),
        "task": "classification",
    },
    # "segmentation": {
    #     "models": ["scalemae_seg.yaml"],
    #     "data_path": os.path.join(
    #         "tests", "configs", "segmentation_dataset_config.yaml"
    #     ),
    #     "task": "segmentation",
    # },
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
    _, model_config = task_and_config
    model_name = model_config.model_type
    model_class = model_registry.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found.")

    # Set environment variable for weights directory
    os.environ["MODEL_WEIGHTS_DIR"] = str(tmp_path)
    model_file_name = os.path.basename(model_config.pretrained_path)
    # Instantiate model without weights
    model_config.pretrained_path = None
    model_without_weights = model_class(other_args, model_config, data_config)
    new_dict = {"model": model_without_weights.state_dict()}

    mocked_path = tmp_path / model_file_name
    torch.save(new_dict, str(mocked_path))
    # Instantiate model with "pretrained" weights
    model_config.pretrained_path = str(mocked_path)
    return model_class(other_args, model_config, data_config)


@pytest.fixture()
def datamodule(data_config):
    return BenchmarkDataModule(
        data_config, num_workers=1, batch_size=2, pin_memory=False
    )


def test_fit(model, datamodule, tmp_path: Path) -> None:
    """Run fit to test the model."""
    # only do for classification
    datamodule.setup()
    if datamodule.dataset_train.task == "classification":
        trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)
        trainer.fit(model, datamodule)


def test_forward_pass(model, datamodule):
    """Test forward pass and loss compute for segmentation."""
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        outputs = model(batch[0])

    loss = model.loss(outputs, batch[1])
    assert loss is not None
    assert torch.isfinite(loss)
