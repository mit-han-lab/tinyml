import torch
import torch.nn as nn

from models.base.layers import (
    ConvLayer,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
)
from utils import make_divisible

__all__ = ["ProxylessNASMobile"]


class ProxylessNASMobile(nn.Module):
    def __init__(
        self, width_mult=1.0, channel_divisor=8, n_classes=1000, dropout_rate=0
    ):
        super(ProxylessNASMobile, self).__init__()
        stage_width_list = [32, 16, 32, 40, 80, 96, 192]
        head_width_list = [320, 1280]
        act_func = "relu6"

        block_configs = [
            [[3, 3], [5, 3], 2, 2],
            [[3, 3, 3, 3], [7, 3, 5, 5], 4, 2],
            [[6, 3, 3, 3], [7, 5, 5, 5], 4, 2],
            [[6, 3, 3, 3], [5, 5, 5, 5], 4, 1],
            [[6, 6, 3, 3], [7, 7, 7, 7], 4, 2],
        ]

        for i, w in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        for i, w in enumerate(head_width_list):
            head_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        head_width_list[1] = max(head_width_list[1], 1280)

        input_stem = OpSequential(
            [
                ConvLayer(3, stage_width_list[0], 3, 2, act_func=act_func),
                ResidualBlock(
                    DsConvLayer(
                        stage_width_list[0],
                        stage_width_list[1],
                        3,
                        1,
                        (act_func, None),
                    ),
                    shortcut=None,
                ),
            ]
        )

        # stages
        stages = []
        in_channels = stage_width_list[1]
        for (e_list, ks_list, n, s), c in zip(block_configs, stage_width_list[2:]):
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                mid_channels = make_divisible(
                    round(e_list[i] * in_channels), channel_divisor
                )
                mb_conv = ResidualBlock(
                    InvertedBlock(
                        in_channels,
                        c,
                        ks_list[i],
                        stride,
                        mid_channels=mid_channels,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=nn.Identity()
                    if (stride == 1 and in_channels == c and i != 0)
                    else None,
                )
                blocks.append(mb_conv)
                in_channels = c
            stages.append(OpSequential(blocks))

        # head
        head = OpSequential(
            [
                ResidualBlock(
                    InvertedBlock(
                        in_channels,
                        head_width_list[0],
                        7,
                        expand_ratio=6,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=None,
                ),
                ConvLayer(head_width_list[0], head_width_list[1], 1, act_func=act_func),
                nn.AdaptiveAvgPool2d(1),
                LinearLayer(head_width_list[1], n_classes, dropout_rate=dropout_rate),
            ]
        )

        self.backbone = nn.ModuleDict(
            {
                "input_stem": input_stem,
                "stages": nn.ModuleList(stages),
            }
        )
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone["input_stem"](x)
        for stage in self.backbone["stages"]:
            x = stage(x)
        x = self.head(x)
        return x
