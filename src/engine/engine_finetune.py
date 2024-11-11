# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy as timm_accuracy
from torchmetrics.functional.classification import multilabel_average_precision, multilabel_f1_score, multiclass_accuracy, binary_accuracy

from torchmetrics.functional import jaccard_index, accuracy
from loguru import logger

import util.misc as misc
import util.lr_sched as lr_sched
import pickle


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if hasattr(model, 'module'):
        task = model.module.task
    else:
        task = model.task

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=False): # disable if loss nan
            match task:
                case 'classification':
                    outputs, feats = model(samples)
                    loss = criterion(outputs, targets)
                case 'segmentation':
                    outputs, outputs_aux = model(samples)
                    loss = criterion(outputs, targets.long()) + 0.4 * criterion(outputs_aux, targets.long())

            '''
            logger.debug("********************Save Features**********************")
            print(outputs.shape, targets.shape)
            feature = feats.cpu().numpy()
            data = {}
            #f_name = f"cross_scale_mae_pv4ger/{data_iter_step}.pkl"
            #f_name = f"cross_scale_mae_brick/{data_iter_step}.pkl"
            f_name = f"croma_so2sat/{data_iter_step}.pkl"
            #f_name = f"cross_scale_mae_eurosat/{data_iter_step}.pkl"
            #f_name = f"cross_scale_mae_forestnet/{data_iter_step}.pkl"
            #f_name = f"cross_scale_mae_bigearthnet/{data_iter_step}.pkl"
            data['feature'] = feature
            data['label'] = targets.cpu().numpy()
            #np.save(f_name, data)
            with open(f_name,'wb') as pfile:
                pickle.dump(data, pfile)
            logger.debug("*******************************************************")
            '''
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cls_metric(dataset_config, output, target):
    if dataset_config.multilabel:
        # Apply sigmoid activation for multilabel classification
        scores = torch.sigmoid(output).detach()

        def precision(average):
            return multilabel_average_precision(
                scores,
                target,
                num_labels=dataset_config.num_classes,
                average=average
            ) * 100
        
        def f1_score(average):
            return multilabel_f1_score(
                scores,
                target,
                num_labels=dataset_config.num_classes,
                average=average
            ) * 100

        return {
            "micro_precision": precision("micro"),
            "macro_precision": precision("macro"),
            "micro_f1": f1_score("micro"),
            "macro_f1": f1_score("macro")
        }
    else:
        # Assuming 'output' contains raw logits 
        if dataset_config.num_classes == 2:
            scores = torch.argmax(output, dim=1)
        else:
            scores = output

        def comp_accuracy(average, topk):
            if dataset_config.num_classes > 2:
                # topk5 needs logits
                return multiclass_accuracy(
                    scores,
                    target,
                    num_classes=dataset_config.num_classes,
                    average=average,
                    top_k=topk
                ) * 100
            else:
                # for binary there is no topk=5, and apply argmax to get the class prediction
                return binary_accuracy(
                    scores,
                    target,
                ) * 100

        return {
            "macro_acc1": comp_accuracy("macro", 1),
            "macro_acc5": comp_accuracy("macro", 5),
            "micro_acc1": comp_accuracy("micro", 1),
            "micro_acc5": comp_accuracy("micro", 5)
        }

def seg_metric(dataset_config, output, target):
    miou = jaccard_index(output, target, task="multiclass", num_classes=dataset_config.num_classes, ignore_index=dataset_config.ignore_index) * 100
    acc = accuracy(output, target, task="multiclass", num_classes=dataset_config.num_classes, ignore_index=dataset_config.ignore_index, top_k=1) * 100
    return {
        "macro_miou": miou,
        "macro_acc": acc
    }


@torch.no_grad()
def evaluate(data_loader, model, device, dataset_config):
    if dataset_config.multilabel:
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    task = dataset_config.task

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            match task:
                case 'classification':
                    output,_ = model(images)
                    loss = criterion(output, target)
                    metrics = cls_metric(dataset_config, output, target)
                case 'segmentation':
                    output, output_aux = model(images)
                    loss = criterion(output, target.long()) + 0.4 * criterion(output_aux, target.long())
                    metrics = seg_metric(dataset_config, output, target)

        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        
        # Update metric_logger with all metrics from the dictionary
        for metric_name, metric_value in metrics.items():
            metric_logger.update(**{metric_name: metric_value}, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Convert metrics to a dictionary for printing
    metrics_dict = {metric: value.global_avg for metric, value in metric_logger.meters.items()}

    print(f'* Metrics: {metrics_dict}')

    return metrics_dict
