from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_act",
    "ConvLayer",
    "LinearLayer",
    "SELayer",
    "DsConvLayer",
    "InvertedBlock",
    "SeInvertedBlock",
    "ResidualBlock",
    "OpSequential",
]


def build_norm(name: Optional[str], num_features: int) -> Optional[nn.Module]:
    if name is None:
        return None
    elif name == "bn_2d":
        return nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError


def build_act(name: Union[str, nn.Module, None]) -> Optional[nn.Module]:
    if name is None:
        return None
    elif isinstance(name, nn.Module):
        return name
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "relu6":
        return nn.ReLU6(inplace=True)
    elif name == "h_swish":
        return nn.Hardswish(inplace=True)
    elif name == "h_sigmoid":
        return nn.Hardsigmoid(inplace=True)
    else:
        raise NotImplementedError


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        norm="bn_2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        padding = kernel_size // 2
        padding *= dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout_rate=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dropout = (
            nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        )
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            for i in range(x.dim() - 1, 1, -1):
                x = torch.squeeze(x, dim=i)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class SELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels=None,
        reduction=4,
        min_dim=16,
        act_func="relu",
    ):
        super(SELayer, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels or max(round(in_channels / reduction), min_dim)
        self.reduction = self.in_channels / self.mid_channels + 1e-10
        self.min_dim = min_dim

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv = nn.Conv2d(
            in_channels, self.mid_channels, kernel_size=(1, 1), bias=True
        )
        self.act = build_act(act_func)
        self.expand_conv = nn.Conv2d(
            self.mid_channels, in_channels, kernel_size=(1, 1), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention = self.pooling(x)
        channel_attention = self.reduce_conv(channel_attention)
        channel_attention = self.act(channel_attention)
        channel_attention = self.expand_conv(channel_attention)
        channel_attention = F.hardsigmoid(channel_attention, inplace=True)
        return x * channel_attention


class DsConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        act_func=("relu6", None),
        norm=("bn_2d", "bn_2d"),
    ):
        super(DsConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class InvertedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
    ):
        super(InvertedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.expand_ratio = self.mid_channels / self.in_channels + 1e-10

        self.inverted_conv = ConvLayer(
            in_channels,
            self.mid_channels,
            1,
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            self.mid_channels,
            self.mid_channels,
            kernel_size,
            stride,
            groups=self.mid_channels,
            norm=norm[1],
            act_func=act_func[1],
        )
        self.point_conv = ConvLayer(
            self.mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class SeInvertedBlock(InvertedBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        act_func=("relu6", "relu6", None),
        norm=("bn_2d", "bn_2d", "bn_2d"),
        se_config: Optional[Dict] = None,
    ):
        super(SeInvertedBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            mid_channels=mid_channels,
            expand_ratio=expand_ratio,
            act_func=act_func,
            norm=norm,
        )
        se_config = se_config or {
            "reduction": 4,
            "min_dim": 16,
            "act_func": "relu",
        }
        self.se_layer = SELayer(self.depth_conv.out_channels, **se_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.se_layer(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, conv: Optional[nn.Module], shortcut: Optional[nn.Module]):
        super(ResidualBlock, self).__init__()
        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return x
        elif self.shortcut is None:
            return self.conv(x)
        else:
            return self.conv(x) + self.shortcut(x)


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
