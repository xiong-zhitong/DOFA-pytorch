import torch
import warnings


# from torch._six import inf
from torch import inf  # pytorch 2.1
import torch.nn.functional as F

from timm.utils import accuracy as timm_accuracy
from torchmetrics.functional.classification import (
    multilabel_average_precision,
    multilabel_f1_score,
)

from torchmetrics.functional import jaccard_index, accuracy


"""This is useful because align_corners=True can cause some artifacts or misalignment, 
   especially if the sizes don’t match in specific ways. This check is not done in F.interpolate.
"""


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def cls_metric(dataset_config, output, target):
    if dataset_config.multilabel:
        score = torch.sigmoid(output).detach()
        acc1 = (
            multilabel_average_precision(
                score, target, num_labels=dataset_config.num_classes, average="micro"
            )
            * 100
        )
        acc5 = (
            multilabel_f1_score(
                score, target, num_labels=dataset_config.num_classes, average="micro"
            )
            * 100
        )
    else:
        acc1, acc5 = timm_accuracy(output, target, topk=(1, 5))
    return acc1, acc5


def seg_metric(dataset_config, output, target):
    miou = (
        jaccard_index(
            output,
            target,
            task="multiclass",
            num_classes=dataset_config.num_classes,
            ignore_index=dataset_config.ignore_index,
        )
        * 100
    )
    acc = (
        accuracy(
            output,
            target,
            task="multiclass",
            num_classes=dataset_config.num_classes,
            ignore_index=dataset_config.ignore_index,
            top_k=1,
        )
        * 100
    )
    return miou, acc
