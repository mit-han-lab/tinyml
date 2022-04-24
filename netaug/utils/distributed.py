from typing import List, Optional, Union

import torch
import torch.distributed
from torchpack import distributed

from utils.misc import list_mean, list_sum

__all__ = ["ddp_reduce_tensor", "DistributedMetric"]


def ddp_reduce_tensor(
    tensor: torch.Tensor, reduce="mean"
) -> Union[torch.Tensor, List[torch.Tensor]]:
    tensor_list = [torch.empty_like(tensor) for _ in range(distributed.size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list


class DistributedMetric(object):
    """Average metrics for distributed training."""

    def __init__(self, name: Optional[str] = None, backend="ddp"):
        self.name = name
        self.sum = 0
        self.count = 0
        self.backend = backend

    def update(self, val: Union[torch.Tensor, int, float], delta_n=1):
        val *= delta_n
        if type(val) in [int, float]:
            val = torch.Tensor(1).fill_(val).cuda()
        if self.backend == "ddp":
            self.count += ddp_reduce_tensor(
                torch.Tensor(1).fill_(delta_n).cuda(), reduce="sum"
            )
            self.sum += ddp_reduce_tensor(val.detach(), reduce="sum")
        else:
            raise NotImplementedError

    @property
    def avg(self):
        if self.count == 0:
            return torch.Tensor(1).fill_(-1)
        else:
            return self.sum / self.count
