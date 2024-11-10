import argparse
import datetime
import json
import pdb
import numpy as np
import os
import time
from pathlib import Path
import warnings
import mlflow

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from loguru import logger

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS

from engine.engine_finetune import train_one_epoch, evaluate

from factory import create_model, create_dataset
from util.config import model_config_registry, dataset_config_registry

def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tune foundation models', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='croma', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Dataset parameters
    parser.add_argument('--dataset', default='geobench_so2sat', type=str, metavar='DATASET',
                        help='Name of model to train')
    
    parser.add_argument('--task', default='segmentation', type=str, metavar='TASK',
                        help='Name of model to train')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument("--local-rank", type=int, default=0)
    
    return parser


def setup_mlflow_experiment(output_dir, model_name, dataset_name):
    """
    Setup MLflow experiment with proper naming and structure
    """
    import mlflow
    import datetime
    from pathlib import Path
    import os

    # Create a meaningful experiment name
    experiment_name = f"{model_name}_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d')}"

    # Set the tracking URI if you want to store MLflow data in a specific location
    mlflow_dir = os.path.join(output_dir, "mlruns")
    Path(mlflow_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")

    # Get or create the experiment explicitly
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Set the experiment context
    mlflow.set_experiment(experiment_name)

    return experiment_id


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset and model
    dataset_config = dataset_config_registry.get(args.dataset)()
    model_config = model_config_registry.get(args.model)()
    model = create_model(model_config, dataset_config)
    model.to(device)

    dataset_train, dataset_val, dataset_test = create_dataset(dataset_config)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.task == "segmentation":
        optimizer = torch.optim.AdamW(model_without_ddp.params_to_optimize(), lr=args.lr)
    elif args.task == "classification":
        optimizer = LARS(model_without_ddp.params_to_optimize(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    if dataset_config.multilabel:
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    # for resume training
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_test, model, device, dataset_config)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_val = 0.0
    max_accuracy_val_test = 0.0

    # Start MLflow run with custom experiment name
    experiment_id = setup_mlflow_experiment(args.output_dir, args.model, dataset_config.dataset_name)
    # Ensure all processes have the same experiment context
    mlflow.set_experiment(experiment_id=experiment_id)

    if args.task == 'classification' and dataset_config.multilabel:
        main_metric = 'macro_f1'
    elif args.task == 'classification':
        main_metric = 'macro_acc1'
    elif args.task == 'segmentation':
        main_metric = 'macro_miou'

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Log all parameters and configs
        mlflow.log_params(vars(args))
        mlflow.log_param("n_parameters", n_parameters)

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                max_norm=None,
                log_writer=None,  # We replace TensorBoard with MLflow
                args=args
            )
            val_stats = evaluate(data_loader_val, model, device, dataset_config)
            test_stats = evaluate(data_loader_test, model, device, dataset_config)

            print(f"Performance of {args.model} on the {len(dataset_val)} val images: {main_metric} - {val_stats[main_metric]:.1f}%")
            print(f"Performance of {args.model} on the {len(dataset_test)} test images: {main_metric} - {test_stats[main_metric]:.1f}%")
                

            if val_stats[main_metric] > max_accuracy_val:
                max_accuracy_val = val_stats[main_metric]
                max_accuracy_val_test = test_stats[main_metric]
                # Save best model checkpoint as an artifact in MLflow
                # Get the MLflow run's artifact directory
                artifact_path = mlflow.get_artifact_uri()
                # Convert the artifact URI to a local file path
                if artifact_path.startswith('file://'):
                    artifact_path = artifact_path[7:]  # Remove 'file://' prefix
        
                # Create models directory in the MLflow artifacts folder
                model_save_dir = os.path.join(artifact_path, "models")
                Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                
                best_epoch_checkpoint = "checkpoint-best"
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=best_epoch_checkpoint, output_dir=model_save_dir)
                    # Verify file exists before logging
                #mlflow.log_artifact(model_checkpoint_path, artifact_path="models")
            print(f'Max val {main_metric}: {max_accuracy_val:.2f}%, test {main_metric}: {max_accuracy_val_test:.2f}%')

            # Log metrics to MLflow with step as the current epoch
            mlflow.log_metric(f'train_loss', train_stats['loss'], step=epoch)
            mlflow.log_metric(f'val_{main_metric}', val_stats[main_metric], step=epoch)
            mlflow.log_metric(f'test_{main_metric}', test_stats[main_metric], step=epoch)
            mlflow.log_metric(f'Max test_{main_metric}', max_accuracy_val_test, step=epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str))
        print('Finished task {}'.format(dataset_config.dataset_name))
        print(f"Max Test {main_metric}: {max_accuracy_val_test}")
        print('******************************************')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
