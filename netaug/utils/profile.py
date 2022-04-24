from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchprofile import profile_macs

__all__ = ["is_parallel", "get_module_device", "trainable_param_num", "inference_macs"]


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def get_module_device(module: nn.Module) -> torch.device:
    return module.parameters().__next__().device


def trainable_param_num(network: nn.Module, unit=1e6) -> float:
    return sum(p.numel() for p in network.parameters() if p.requires_grad) / unit


def inference_macs(
    network: nn.Module,
    args: Tuple = (),
    data_shape: Optional[Tuple] = None,
    unit: float = 1e6,
) -> float:
    if is_parallel(network):
        network = network.module
    if data_shape is not None:
        if len(args) > 0:
            raise ValueError("Please provide either data_shape or args tuple.")
        args = (torch.zeros(data_shape, device=get_module_device(network)),)
    is_training = network.training
    network.eval()
    macs = profile_macs(network, args=args) / unit
    network.train(is_training)
    return macs
