import argparse
import os
import datetime
from lightning import Trainer, seed_everything
from typing import Any, Dict
from pathlib import Path

import ray

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)

import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train import RunConfig

# import ray
# from ray.tune.schedulers import MedianStoppingRule
# from ray.tune.search.bayesopt import BayesOptSearch
# from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

# Your existing project imports
from config import model_config_registry, dataset_config_registry
from datasets.data_module import BenchmarkDataModule
from factory import create_model


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RayTune for Hyperparameter Tuning", add_help=False
    )
    parser.add_argument("--model", default="croma", type=str)
    parser.add_argument("--dataset", default="geobench_so2sat", type=str)
    parser.add_argument("--task", default="segmentation", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # training args for a particular single trial
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_gpus", type=int, default=1)  # needed to compute schedule
    parser.add_argument("--warmup_epochs", type=int, default=3)

    # Hyperparameter search ranges
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--batch_choices", type=int, nargs="+", default=[16, 32, 64])

    # RayTune parameters
    parser.add_argument("--ray_max_concurrent_trials", type=int, default=4)
    parser.add_argument("--ray_num_samples", type=int, default=10)

    # Logging / Output
    parser.add_argument("--output_dir", default="./raytune_logs")
    return parser


def train_with_tune(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Train function that integrates with Ray Tune to run a single trial.

    Args:
        config (dict): A dictionary containing hyperparameters sampled by Ray Tune
                       (e.g., {'lr': 1e-4, 'batch_size': 32}). This will be passed by
                       RayTune and includes the parameters specified in the search space
                       with specific values for this trial
        args (Namespace): Additional arguments used to build the model and dataset
    """
    seed_everything(args.seed)

    args.lr = config["lr"]
    args.batch_size = config["batch_size"]

    # Load dataset + model configs
    dataset_config = dataset_config_registry.get(args.dataset)()
    model_config = model_config_registry.get(args.model)()

    data_module = BenchmarkDataModule(
        dataset_config=dataset_config,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    model = create_model(args, model_config, dataset_config)

    # Create Trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        accelerator="auto",
        devices="auto",
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)

    trainer.fit(model, data_module)


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = args.output_dir + "_" + timestamp
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    scaling_config = ray.train.ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}
    )
    run_config = RunConfig(storage_path=args.output_dir, name="unique_run_name")

    ray_trainer = TorchTrainer(
        # tune.with_parameters allows passing 'args' or additional params to train_with_tune
        tune.with_parameters(train_with_tune, args=args),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    # Define the search space
    search_space = {
        "lr": tune.loguniform(args.lr_min, args.lr_max),
        "batch_size": tune.choice(args.batch_choices),
    }

    # Use an ASHAScheduler for early stopping of unpromising trials
    # grace period is number of epochs before trial is stopped
    scheduler = ASHAScheduler(max_t=args.epochs, grace_period=5)

    model_monitor = "val_miou" if args.task == "segmentation" else "val_acc1"

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric=model_monitor,
            mode="max",
            num_samples=args.ray_num_samples,
            scheduler=scheduler,
            search_alg=OptunaSearch(metric=model_monitor, mode="max"),
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
