import copy
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from tqdm import tqdm

from models.base.layers import InvertedBlock, SeInvertedBlock, SELayer
from utils.distributed import ddp_reduce_tensor
from utils.metric import AverageMeter
from utils.misc import make_divisible

__all__ = ["reset_bn", "aug_width", "sync_width", "sort_param", "sort_channels_inner"]


def reset_bn(
    model: nn.Module, data_loader, sync=False, backend="ddp", progress_bar=False
) -> None:
    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter()
            bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        if backend == "ddp":
                            batch_mean = ddp_reduce_tensor(batch_mean, reduce="cat")
                        else:
                            raise NotImplementedError
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )
                        if backend == "ddp":
                            batch_var = ddp_reduce_tensor(batch_var, reduce="cat")
                        else:
                            raise NotImplementedError
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = (
                            x.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = (
                            batch_var.mean(0, keepdim=True)
                            .mean(2, keepdim=True)
                            .mean(3, keepdim=True)
                        )

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(data_loader), desc="reset bn", disable=(not progress_bar)
        ) as t:
            for images, _ in data_loader:
                images = images.cuda()
                tmp_model(images)
                t.set_postfix(
                    {
                        "batch_size": images.size(0),
                        "image_size": images.size(2),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def aug_width(
    base_width: float, factor_list: List[float], divisor: Optional[int] = None
) -> List[Union[float, int]]:
    out_list = [base_width * factor for factor in factor_list]
    if divisor is not None:
        out_list = [make_divisible(out_dim, divisor) for out_dim in out_list]
    return out_list


def sync_width(width) -> int:
    width = ddp_reduce_tensor(torch.Tensor(1).fill_(width).cuda(), "root")
    return int(width)


def sort_param(
    param: nn.Parameter,
    dim: int,
    sorted_idx: torch.Tensor,
) -> None:
    param.data.copy_(
        torch.clone(torch.index_select(param.data, dim, sorted_idx)).detach()
    )


def sort_norm(norm, sorted_idx: torch.Tensor) -> None:
    sort_param(norm.weight, 0, sorted_idx)
    sort_param(norm.bias, 0, sorted_idx)
    try:
        sort_param(norm.running_mean, 0, sorted_idx)
        sort_param(norm.running_var, 0, sorted_idx)
    except AttributeError:
        pass


def sort_se(se: SELayer, sorted_idx: torch.Tensor) -> None:
    # expand conv, output dim 0
    sort_param(se.expand_conv.weight, 0, sorted_idx)
    sort_param(se.expand_conv.bias, 0, sorted_idx)
    # reduce conv, input dim 1
    sort_param(se.reduce_conv.weight, 1, sorted_idx)

    # sort middle weight
    importance = torch.sum(torch.abs(se.expand_conv.weight.data), dim=(0, 2, 3))
    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    # expand conv, input dim 1
    sort_param(se.expand_conv.weight, 1, sorted_idx)
    # reduce conv, output dim 0
    sort_param(se.reduce_conv.weight, 0, sorted_idx)
    sort_param(se.reduce_conv.bias, 0, sorted_idx)


def sort_channels_inner(block) -> None:
    if isinstance(block, (InvertedBlock, SeInvertedBlock)):
        # calc channel importance
        importance = torch.sum(
            torch.abs(block.point_conv.conv.weight.data), dim=(0, 2, 3)
        )
        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        # sort based on sorted_idx
        sort_param(block.point_conv.conv.weight, 1, sorted_idx)
        sort_norm(block.depth_conv.norm, sorted_idx)
        sort_param(block.depth_conv.conv.weight, 0, sorted_idx)
        sort_norm(block.inverted_conv.norm, sorted_idx)
        sort_param(block.inverted_conv.conv.weight, 0, sorted_idx)
        if isinstance(block, SeInvertedBlock):
            sort_se(block.se_layer, sorted_idx)
    else:
        raise NotImplementedError
