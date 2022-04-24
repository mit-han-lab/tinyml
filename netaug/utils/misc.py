from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = [
    "make_divisible",
    "load_state_dict_from_file",
    "list_mean",
    "list_sum",
    "parse_unknown_args",
    "partial_update_config",
    "remove_bn",
    "get_same_padding",
    "torch_random_choices",
]


def make_divisible(
    v: Union[int, float], divisor: Optional[int], min_val=None
) -> Union[int, float]:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if divisor is None:
        return v

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def load_state_dict_from_file(file: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(file, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: List) -> Any:
    return list_sum(x) / len(x)


def parse_unknown_args(unknown: List) -> Dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key, val = unknown[index], unknown[index + 1]
        index += 2
        if key.startswith("--"):
            key = key[2:]
            try:
                # try parsing with yaml
                if "{" in val and "}" in val and ":" in val:
                    val = val.replace(":", ": ")  # add space manually for dict
                out_val = yaml.safe_load(val)
            except ValueError:
                # return raw string if parsing fails
                out_val = val
            parsed_dict[key] = out_val
    return parsed_dict


def partial_update_config(config: Dict, partial_config: Dict):
    for key in partial_config:
        if (
            key in config
            and isinstance(partial_config[key], Dict)
            and isinstance(config[key], Dict)
        ):
            partial_update_config(config[key], partial_config[key])
        else:
            config[key] = partial_config[key]


def remove_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.weight = m.bias = None
            m.forward = lambda x: x


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, tuple]:
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(
            kernel_size, int
        ), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def torch_random_choices(
    src_list: List[Any],
    generator: Optional[torch.Generator],
    k=1,
) -> Union[Any, List[Any]]:
    rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
    out_list = [src_list[i] for i in rand_idx]
    return out_list[0] if k == 1 else out_list
