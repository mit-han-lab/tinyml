from typing import List, Union

import numpy as np
import torch

__all__ = ["accuracy", "AverageMeter"]


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1,)
) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value.

    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[torch.Tensor, np.ndarray, float, int], n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
