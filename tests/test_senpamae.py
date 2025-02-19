import pytest
import os
import argparse
from pathlib import Path
import torch


from geofm_src.factory import model_registry
from geofm_src.datasets.data_module import BenchmarkDataModule
from omegaconf import OmegaConf
from lightning import Trainer
from hydra import compose, initialize

classification_configs = ["senpamae_cls", "senpamae_cls_selective_params"]


class TestClassificationSenpaMAE:
    @pytest.fixture()
    def other_args(self):
        args = argparse.Namespace()
        args.task = "classification"
        args.lr = 0.001
        args.weight_decay = 0.0
        args.warmup_epochs = 0
        args.num_gpus = 0
        args.epochs = 1
        return args

    @pytest.fixture(
        params=classification_configs,
    )
    def model_config(self, request):
        with initialize(
            version_base=None, config_path=os.path.join("..", "src", "configs")
        ):
            model_config = compose(
                config_name="config", overrides=[f"model={request.param}"]
            )

        return model_config.model

    @pytest.fixture(
        params=[
            os.path.join("src", "configs", "dataset", "s2_l1c.yaml"),
            os.path.join("src", "configs", "dataset", "s2_l2a.yaml"),
        ]
    )
    def data_config(self, model_config, request):
        data_config_path = os.path.join(
            "tests", "configs", "classification_dataset_config.yaml"
        )
        data_config = OmegaConf.load(data_config_path)

        if "image_resolution" in model_config:
            data_config.image_resolution = model_config.image_resolution

        if "num_channels" in model_config:
            data_config.num_channels = model_config.num_channels

        extra_config = OmegaConf.load(request.param)

        data_config = OmegaConf.merge(data_config, extra_config)

        return data_config

    @pytest.fixture
    def model(
        self,
        model_config,
        other_args,
        data_config,
        tmp_path: Path,
    ):
        model_name = model_config.model_type
        model_class = model_registry.get(model_name)
        if model_class is None:
            raise ValueError(f"Model type '{model_name}' not found.")

        # Set environment variable for weights directory
        os.environ["MODEL_WEIGHTS_DIR"] = str(tmp_path)
        model_file_name = os.path.basename(model_config.pretrained_path)

        model_config.pretrained_path = None
        # instantiate model without pretrained_path
        model_without_weights = model_class(other_args, model_config, data_config)

        new_dict = {"model": model_without_weights.state_dict()}

        mocked_path = tmp_path / model_file_name

        torch.save(new_dict, str(mocked_path))

        # instantiate with pretraine_path
        model_config.pretrained_path = str(mocked_path)

        model_with_weights = model_class(other_args, model_config, data_config)
        return model_with_weights

    @pytest.fixture()
    def datamodule(self, data_config):
        return BenchmarkDataModule(
            data_config, num_workers=1, batch_size=2, pin_memory=False
        )

    def test_fit(self, model, datamodule, tmp_path: Path) -> None:
        """Test lightning fit."""

        trainer = Trainer(max_epochs=1, default_root_dir=tmp_path, log_every_n_steps=1)

        trainer.fit(model, datamodule)
