import datetime
import os
from pathlib import Path
from typing import Dict, Any

import hydra
from omegaconf import DictConfig
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger

from datasets.data_module import BenchmarkDataModule
from factory import create_model


def train_with_tune(config: Dict[str, Any], hydra_config: DictConfig) -> None:
    """Train function for a single Ray Tune trial using Hydra config."""
    seed_everything(hydra_config.seed)

    # Update config with trial-specific hyperparameters
    hydra_config.lr = config["lr"]
    hydra_config.batch_size = config["batch_size"]

    # Setup logging
    experiment_name = (
        f"{hydra_config.model.model_type}_{hydra_config.dataset.dataset_name}"
    )
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_trial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tracking_uri=f"file:{os.path.join(hydra_config.output_dir, 'mlruns')}",
    )

    # Initialize data and model
    data_module = BenchmarkDataModule(
        dataset_config=hydra_config.dataset,
        batch_size=config["batch_size"],
        num_workers=hydra_config.num_workers,
        pin_memory=hydra_config.pin_mem,
    )
    model = create_model(hydra_config, hydra_config.model, hydra_config.dataset)

    # Create trainer with Ray-specific settings
    trainer = Trainer(
        max_epochs=hydra_config.epochs,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        logger=mlf_logger,
        accelerator="auto",
        devices="auto",
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, data_module)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function combining Hydra and Ray"""

    # Setup directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = f"{cfg.output_dir}_{timestamp}"
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Ray configuration
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={
            "CPU": cfg.ray.get("cpus_per_trial", 4),
            "GPU": cfg.ray.get("gpus_per_trial", 1),
        },
    )

    run_config = RunConfig(
        storage_path=cfg.output_dir,
        name=f"tune_{cfg.model.model_type}_{cfg.dataset.dataset_name}",
    )

    ray_trainer = TorchTrainer(
        tune.with_parameters(train_with_tune, hydra_config=cfg),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # Search space from args
    search_space = {
        "lr": tune.loguniform(cfg.ray.lr_min, cfg.ray.lr_max),
        "batch_size": tune.choice(cfg.ray.batch_choices),
    }

    model_monitor = "val_miou" if cfg.task == "segmentation" else "val_acc1"
    # Setup tuner
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric=model_monitor,
            mode="max",
            num_samples=cfg.ray.get("num_samples", 10),
            scheduler=ASHAScheduler(
                max_t=cfg.epochs, grace_period=cfg.ray.get("grace_period", 5)
            ),
            search_alg=OptunaSearch(),
        ),
    )

    results = tuner.fit()

    best_trial = results.get_best_result(model_monitor, "max")

    # write to file
    best_trial_file = os.path.join(args.output_dir, "best_trial.txt")
    with open(best_trial_file, "w") as f:
        f.write(f"Best trial config: {best_trial.config}\n")
        f.write(f"Best trial final metric: {best_trial[model_monitor]}\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
