import copy
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _ntuple

from models.base import (
    ConvLayer,
    DsConvLayer,
    InvertedBlock,
    LinearLayer,
    SeInvertedBlock,
    SELayer,
    build_act,
)
from utils import get_same_padding, make_divisible

__all__ = [
    "build_norm",
    "DynamicModule",
    "DynamicConv2d",
    "DynamicDepthwiseConv2d",
    "DynamicLinear",
    "DynamicBatchNorm2d",
    "DynamicConvLayer",
    "DynamicDepthwiseConvLayer",
    "DynamicLinearLayer",
    "DynamicSE",
    "DynamicDsConvLayer",
    "DynamicInvertedBlock",
    "DynamicSeInvertedBlock",
]


def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn_2d":
        return DynamicBatchNorm2d(num_features)
    else:
        raise NotImplementedError


class DynamicModule(nn.Module):
    def export(self) -> nn.Module:
        raise NotImplementedError

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = self.state_dict()
        for prefix, module in self.named_children():
            if isinstance(module, DynamicModule):
                for name, tensor in module.active_state_dict().items():
                    state_dict[prefix + "." + name] = tensor
        return state_dict


class DynamicConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=1,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=1,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=1,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicDepthwiseConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=_ntuple(self._ndim)(0),
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_in_channels, : self.active_in_channels]
        return weight.contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_in_channels].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            active_weight = self.active_weight
            return getattr(F, "conv{}d".format(self._ndim))(
                x,
                active_weight,
                self.active_bias,
                stride=self.stride,
                padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                dilation=self.dilation,
                groups=self.active_in_channels,
            )

    def export(self) -> nn.Module:
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_in_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            dilation=self.dilation,
            groups=self.active_in_channels,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicLinear(nn.Linear, DynamicModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.active_in_features = in_features
        self.active_out_features = out_features

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[
            : self.active_out_features, : self.active_in_features
        ].contiguous()

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_features].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_features = x.shape[-1]
        return F.linear(x, weight=self.active_weight, bias=self.active_bias)

    def export(self) -> nn.Module:
        module = nn.Linear(
            self.active_in_features,
            self.active_out_features,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicBatchNorm2d(DynamicModule, nn.BatchNorm2d):
    _ndim = 2

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        nn.BatchNorm2d.__init__(
            self,
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.active_num_features = num_features

    @property
    def active_running_mean(self) -> Optional[torch.Tensor]:
        if self.running_mean is None:
            return None
        return self.running_mean[: self.active_num_features]

    @property
    def active_running_var(self) -> Optional[torch.Tensor]:
        if self.running_var is None:
            return None
        return self.running_var[: self.active_num_features]

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[: self.active_num_features]

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_num_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        self.active_num_features = x.shape[1]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.active_running_mean is None) and (
                self.active_running_var is None
            )

        running_mean = (
            self.active_running_mean
            if not self.training or self.track_running_stats
            else None
        )
        running_var = (
            self.active_running_var
            if not self.training or self.track_running_stats
            else None
        )

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.active_weight,
            self.active_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def export(self) -> nn.Module:
        module = getattr(nn, "BatchNorm{}d".format(self._ndim))(
            self.active_num_features,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.running_mean is not None:
            state_dict["running_mean"] = self.active_running_mean
        if self.running_var is not None:
            state_dict["running_var"] = self.active_running_var
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicConvLayer(ConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicConv2d(
            in_channels=in_channels,
            out_channels=max(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, max(out_channels))
        self.act = build_act(act_func)

        self.in_channels = in_channels
        self.out_channels_list = copy.deepcopy(out_channels)

    def export(self) -> ConvLayer:
        module = ConvLayer.__new__(ConvLayer)
        nn.Module.__init__(module)
        module.conv = self.conv.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module


class DynamicDepthwiseConvLayer(DynamicConvLayer):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride=1,
        dilation=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        nn.Module.__init__(self)
        self.conv = DynamicDepthwiseConv2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias,
        )
        self.norm = build_norm(norm, in_channels)
        self.act = build_act(act_func)


class DynamicLinearLayer(LinearLayer, DynamicModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        DynamicModule.__init__(self)

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = DynamicLinear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, out_features)
        self.act = build_act(act_func)

    def export(self) -> LinearLayer:
        module = LinearLayer.__new__(LinearLayer)
        nn.Module.__init__(module)
        module.dropout = self.dropout
        module.linear = self.linear.export()
        module.norm = (
            self.norm.export() if isinstance(self.norm, DynamicModule) else self.norm
        )
        module.act = self.act
        return module


class DynamicSE(SELayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        DynamicModule.__init__(self)
        self.min_dim = min_dim

        if mid_channels is None:
            mid_channels = max(round(in_channels / reduction), min_dim)
        self.reduction = in_channels / mid_channels

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = DynamicConv2d(
            in_channels, mid_channels, kernel_size=1, bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = DynamicConv2d(
            mid_channels, in_channels, kernel_size=1, bias=True
        )

        self.active_in_channels = in_channels

    @property
    def active_mid_channels(self):
        return make_divisible(
            max(self.active_in_channels / self.reduction, self.min_dim), 1
        )

    def forward(self, x):
        self.active_in_channels = x.shape[1]
        self.reduce_conv.active_in_channels = self.active_in_channels
        self.reduce_conv.active_out_channels = self.active_mid_channels
        self.expand_conv.active_in_channels = self.active_mid_channels
        self.expand_conv.active_out_channels = self.active_in_channels

        return SELayer.forward(self, x)

    def export(self) -> SELayer:
        module = SELayer(
            in_channels=self.active_in_channels,
            mid_channels=self.active_mid_channels,
        )
        module.act = self.act
        module.load_state_dict(self.active_state_dict())
        return module


class DynamicDsConvLayer(DsConvLayer, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def export(self) -> DsConvLayer:
        module = DsConvLayer.__new__(DsConvLayer)
        nn.Module.__init__(module)
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicInvertedBlock(InvertedBlock, DynamicModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        nn.Module.__init__(self)

        mid_channels = make_divisible(in_channels * max(expand_ratio), 1)

        self.inverted_conv = DynamicConvLayer(
            in_channels=in_channels,
            out_channels=[mid_channels],
            kernel_size=1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = DynamicDepthwiseConvLayer(
            in_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = DynamicConvLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm=norm[2],
            act_func=act_func[2],
        )

        self.expand_ratio_list = copy.deepcopy(expand_ratio)

    def export(self) -> InvertedBlock:
        module = InvertedBlock.__new__(InvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.point_conv = self.point_conv.export()
        return module


class DynamicSeInvertedBlock(SeInvertedBlock, DynamicInvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: List[int],
        kernel_size: int,
        expand_ratio: List[float],
        stride=1,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        DynamicInvertedBlock.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            stride=stride,
            act_func=act_func,
            norm=norm,
        )

        se_config = {} or se_config
        self.se_layer = DynamicSE(in_channels=self.point_conv.in_channels, **se_config)

    def export(self) -> SeInvertedBlock:
        module = SeInvertedBlock.__new__(SeInvertedBlock)
        nn.Module.__init__(module)
        module.inverted_conv = self.inverted_conv.export()
        module.depth_conv = self.depth_conv.export()
        module.se_layer = self.se_layer.export()
        module.point_conv = self.point_conv.export()
        return module
