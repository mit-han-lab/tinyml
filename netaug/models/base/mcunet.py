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

__all__ = ["MCUNet"]


class MCUNet(nn.Module):
    def __init__(self, channel_divisor=8, n_classes=1000, dropout_rate=0):
        super(MCUNet, self).__init__()
        stage_width_list = [16, 8, 16, 24, 40, 48, 96]
        head_width_list = [160]
        act_func = "relu6"

        block_configs = [
            [[3, 5, 5, 4], [7, 3, 7, 5], 4, 2],
            [[5, 5, 5], [5, 5, 5], 3, 2],
            [[5, 6, 4], [3, 7, 5], 3, 2],
            [[5, 5, 5], [5, 7, 3], 3, 1],
            [[6, 5, 4], [3, 7, 3], 3, 2],
        ]

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
                        mid_channels=480,
                        act_func=(act_func, act_func, None),
                    ),
                    shortcut=None,
                ),
                nn.AdaptiveAvgPool2d(1),
                LinearLayer(head_width_list[0], n_classes, dropout_rate=dropout_rate),
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
