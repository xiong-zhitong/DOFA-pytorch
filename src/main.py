import argparse
import datetime
import os
from pathlib import Path
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets.data_module import LightningDataModule

from factory import create_model
from model_config import model_config_registry
from dataset_config import dataset_config_registry

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("Fine-tune foundation models", add_help=False)

    # Data args
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # Model parameters
    parser.add_argument("--model", default="croma", type=str, metavar="MODEL")
    parser.add_argument(
        "--dataset", default="geobench_so2sat", type=str, metavar="DATASET"
    )
    parser.add_argument("--task", default="segmentation", type=str, metavar="TASK")

    # Optimizer parameters
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Output parameters
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")

    return parser


def main(args):
    pl.seed_everything(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup configs
    dataset_config = dataset_config_registry.get(args.dataset)()
    model_config = model_config_registry.get(args.model)()

    args.lr = args.lr * args.num_gpus

    experiment_name = f"{args.model}_{args.dataset}"
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tracking_uri=f"file:{os.path.join(args.output_dir, 'mlruns')}",
    )

    # Callbacks
    model_monitor = "val_miou" if args.task == "segmentation" else "val_acc1"
    # model_monitor = "val_loss"
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best_model-{epoch}",
            monitor=model_monitor,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
        devices="auto",
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    # Initialize data module
    data_module = LightningDataModule(
        dataset_config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    # Create model (assumed to be a LightningModule)
    model = create_model(args, model_config, dataset_config)

    # Train
    trainer.fit(model, data_module, ckpt_path=args.resume if args.resume else None)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    trainer.test(model, data_module, ckpt_path=best_checkpoint_path)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
