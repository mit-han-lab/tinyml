import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["init_modules", "load_state_dict"]


def init_modules(
    module: Union[nn.Module, List[nn.Module]], init_type="he_fout"
) -> None:
    init_params = init_type.split("@")
    if len(init_params) > 1:
        init_params = float(init_params[1])
    else:
        init_params = None

    if isinstance(module, list):
        for sub_module in module:
            init_modules(sub_module)
    else:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == "he_fout":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif init_type.startswith("kaiming_uniform"):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(init_params or 5))
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(init_params or 5))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            else:
                weight = getattr(m, "weight", None)
                bias = getattr(m, "bias", None)
                if isinstance(weight, torch.nn.Parameter):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(init_params or 5))
                if isinstance(bias, torch.nn.Parameter):
                    bias.data.zero_()


def load_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], strict=True
) -> None:
    current_state_dict = model.state_dict()
    for key in state_dict:
        if current_state_dict[key].shape != state_dict[key].shape:
            if strict:
                raise ValueError(
                    "%s shape mismatch (src=%s, target=%s)"
                    % (
                        key,
                        list(state_dict[key].shape),
                        list(current_state_dict[key].shape),
                    )
                )
            else:
                print(
                    "Skip loading %s due to shape mismatch (src=%s, target=%s)"
                    % (
                        key,
                        list(state_dict[key].shape),
                        list(current_state_dict[key].shape),
                    )
                )
        else:
            current_state_dict[key].copy_(state_dict[key])
    model.load_state_dict(current_state_dict)
