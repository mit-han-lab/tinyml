from typing import List

from models.base.tiny_mbv2 import TinyMobileNetV2

from .mbv2 import NetAugMobileNetV2

__all__ = ["NetAugTinyMobileNetV2"]


class NetAugTinyMobileNetV2(NetAugMobileNetV2):
    def __init__(
        self,
        base_net: TinyMobileNetV2,
        aug_expand_list: List[float],
        aug_width_mult_list: List[float],
        n_classes: int,
        dropout_rate=0.0,
    ):
        super(NetAugTinyMobileNetV2, self).__init__(
            base_net,
            aug_expand_list,
            aug_width_mult_list,
            n_classes,
            dropout_rate,
        )
