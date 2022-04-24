from typing import List

from models.base.proxylessnas import ProxylessNASMobile

from .mbv2 import NetAugMobileNetV2

__all__ = ["NetAugProxylessNASMobile"]


class NetAugProxylessNASMobile(NetAugMobileNetV2):
    def __init__(
        self,
        base_net: ProxylessNASMobile,
        aug_expand_list: List[float],
        aug_width_mult_list: List[float],
        n_classes: int,
        dropout_rate=0.0,
    ):
        super(NetAugProxylessNASMobile, self).__init__(
            base_net,
            aug_expand_list,
            aug_width_mult_list,
            n_classes,
            dropout_rate,
        )
