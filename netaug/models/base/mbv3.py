import torch
import torch.nn as nn

from models.base.layers import (
    ConvLayer,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
    SeInvertedBlock,
)
from utils import make_divisible

__all__ = ["MobileNetV3"]


class MobileNetV3(nn.Module):
    def __init__(
        self, width_mult=1.0, channel_divisor=8, n_classes=1000, dropout_rate=0
    ):
        super(MobileNetV3, self).__init__()
        stage_width_list = [16, 24, 40, 80, 112, 160]
        head_width_list = [960, 1280]

        block_configs = [
            [[64, 72], 3, 2, 2, "relu", False],
            [[72, 120, 120], 5, 3, 2, "relu", True],
            [[240, 200, 184, 184], 3, 4, 2, "h_swish", False],
            [[480, 672], 3, 2, 1, "h_swish", True],
            [[672, 960, 960], 5, 3, 2, "h_swish", True],
        ]

        for i, w in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        for i, w in enumerate(head_width_list):
            head_width_list[i] = make_divisible(w * width_mult, channel_divisor)
        head_width_list[1] = max(head_width_list[1], 1280)

        input_stem = OpSequential(
            [
                ConvLayer(3, stage_width_list[0], 3, 2, act_func="h_swish"),
                ResidualBlock(
                    DsConvLayer(
                        stage_width_list[0],
                        stage_width_list[0],
                        3,
                        1,
                        ("relu", None),
                    ),
                    shortcut=nn.Identity(),
                ),
            ]
        )

        # stages
        stages = []
        in_channels = stage_width_list[0]
        for (mid_c_list, ks, n, s, act_func, use_se), c in zip(
            block_configs, stage_width_list[1:]
        ):
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                mid_channels = make_divisible(
                    round(mid_c_list[i] * width_mult), channel_divisor
                )
                if use_se:
                    conv = SeInvertedBlock(
                        in_channels,
                        c,
                        ks,
                        stride,
                        mid_channels=mid_channels,
                        act_func=(act_func, act_func, None),
                        se_config={
                            "act_func": "relu",
                            "mid_channels": max(
                                make_divisible(mid_channels / 4, channel_divisor), 16
                            ),
                        },
                    )
                else:
                    conv = InvertedBlock(
                        in_channels,
                        c,
                        ks,
                        stride,
                        mid_channels=mid_channels,
                        act_func=(act_func, act_func, None),
                    )
                mb_conv = ResidualBlock(
                    conv,
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
                ConvLayer(in_channels, head_width_list[0], 1, act_func="h_swish"),
                nn.AdaptiveAvgPool2d(1),
                ConvLayer(
                    head_width_list[0],
                    head_width_list[1],
                    1,
                    act_func="h_swish",
                    norm=None,
                    use_bias=True,
                ),
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
