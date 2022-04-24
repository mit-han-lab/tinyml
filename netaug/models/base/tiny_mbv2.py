import torch.nn as nn

from models.base.layers import (
    ConvLayer,
    InvertedBlock,
    LinearLayer,
    OpSequential,
    ResidualBlock,
)
from models.base.mbv2 import MobileNetV2


class TinyMobileNetV2(MobileNetV2):
    def __init__(self, channel_divisor=8, n_classes=1000, dropout_rate=0):
        super(TinyMobileNetV2, self).__init__(
            0.35, channel_divisor, n_classes, dropout_rate
        )

        self.head = OpSequential(
            [
                ResidualBlock(
                    InvertedBlock(
                        56,
                        112,
                        3,
                        expand_ratio=6,
                        act_func=("relu6", "relu6", None),
                    ),
                    shortcut=None,
                ),
                ConvLayer(112, 448, 1, act_func="relu6"),
                nn.AdaptiveAvgPool2d(1),
                LinearLayer(448, n_classes, dropout_rate=dropout_rate),
            ]
        )
