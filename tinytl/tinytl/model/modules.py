import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from ofa.utils.layers import set_layer_from_config, ZeroLayer
from ofa.utils import MyModule, MyNetwork, MyGlobalAvgPool2d, min_divisible_value, SEModule
from ofa.utils import get_same_padding, make_divisible, build_activation, init_models

__all__ = ['my_set_layer_from_config',
           'LiteResidualModule', 'ReducedMBConvLayer']


def my_set_layer_from_config(layer_config):
	if layer_config is None:
		return None
	name2layer = {
		LiteResidualModule.__name__: LiteResidualModule,
		ReducedMBConvLayer.__name__: ReducedMBConvLayer,
	}
	layer_name = layer_config.pop('name')
	if layer_name in name2layer:
		layer = name2layer[layer_name]
		return layer.build_from_config(layer_config)
	else:
		return set_layer_from_config({'name': layer_name, **layer_config})


class LiteResidualModule(MyModule):

	def __init__(self, main_branch, in_channels, out_channels,
	             expand=1.0, kernel_size=3, act_func='relu', n_groups=2,
	             downsample_ratio=2, upsample_type='bilinear', stride=1):
		super(LiteResidualModule, self).__init__()

		self.main_branch = main_branch

		self.lite_residual_config = {
			'in_channels': in_channels,
			'out_channels': out_channels,
			'expand': expand,
			'kernel_size': kernel_size,
			'act_func': act_func,
			'n_groups': n_groups,
			'downsample_ratio': downsample_ratio,
			'upsample_type': upsample_type,
			'stride': stride,
		}

		kernel_size = 1 if downsample_ratio is None else kernel_size

		padding = get_same_padding(kernel_size)
		if downsample_ratio is None:
			pooling = MyGlobalAvgPool2d()
		else:
			pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)
		num_mid = make_divisible(int(in_channels * expand), divisor=MyNetwork.CHANNEL_DIVISIBLE)
		self.lite_residual = nn.Sequential(OrderedDict({
			'pooling': pooling,
			'conv1': nn.Conv2d(in_channels, num_mid, kernel_size, stride, padding, groups=n_groups, bias=False),
			'bn1': nn.BatchNorm2d(num_mid),
			'act': build_activation(act_func),
			'conv2': nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
			'final_bn': nn.BatchNorm2d(out_channels),
		}))

		# initialize
		init_models(self.lite_residual)
		self.lite_residual.final_bn.weight.data.zero_()

	def forward(self, x):
		main_x = self.main_branch(x)
		lite_residual_x = self.lite_residual(x)
		if self.lite_residual_config['downsample_ratio'] is not None:
			lite_residual_x = F.upsample(lite_residual_x, main_x.shape[2:],
			                             mode=self.lite_residual_config['upsample_type'])
		return main_x + lite_residual_x

	@property
	def module_str(self):
		return self.main_branch.module_str + ' + LiteResidual(downsample=%s, n_groups=%s, expand=%s, ks=%s)' % (
			self.lite_residual_config['downsample_ratio'], self.lite_residual_config['n_groups'],
			self.lite_residual_config['expand'], self.lite_residual_config['kernel_size'],
		)

	@property
	def config(self):
		return {
			'name': LiteResidualModule.__name__,
			'main': self.main_branch.config,
			'lite_residual': self.lite_residual_config,
		}

	@staticmethod
	def build_from_config(config):
		main_branch = my_set_layer_from_config(config['main'])
		lite_residual_module = LiteResidualModule(
			main_branch, **config['lite_residual']
		)
		return lite_residual_module

	def __repr__(self):
		return '{\n (main branch): ' + self.main_branch.__repr__() + ', ' + \
		       '\n (lite residual): ' + self.lite_residual.__repr__() + '}'

	@staticmethod
	def insert_lite_residual(net, downsample_ratio=2, upsample_type='bilinear',
	                         expand=1.0, max_kernel_size=5, act_func='relu', n_groups=2,
	                         **kwargs):
		if LiteResidualModule.has_lite_residual_module(net):
			# skip if already has lite residual modules
			return
		from ofa.imagenet_classification.networks import ProxylessNASNets
		if isinstance(net, ProxylessNASNets):
			bn_param = net.get_bn_param()

			# blocks
			max_resolution = 128
			stride_stages = [2, 2, 2, 1, 2, 1]
			for block_index_list, stride in zip(net.grouped_block_index, stride_stages):
				for i, idx in enumerate(block_index_list):
					block = net.blocks[idx].conv
					if isinstance(block, ZeroLayer):
						continue
					s = stride if i == 0 else 1
					block_downsample_ratio = downsample_ratio
					block_resolution = max(1, max_resolution // block_downsample_ratio)
					max_resolution //= s

					kernel_size = max_kernel_size
					if block_resolution == 1:
						kernel_size = 1
						block_downsample_ratio = None
					else:
						while block_resolution < kernel_size:
							kernel_size -= 2
					net.blocks[idx].conv = LiteResidualModule(
						block, block.in_channels, block.out_channels, expand=expand, kernel_size=kernel_size,
						act_func=act_func, n_groups=n_groups, downsample_ratio=block_downsample_ratio,
						upsample_type=upsample_type, stride=s,
					)

			net.set_bn_param(**bn_param)
		else:
			raise NotImplementedError

	@staticmethod
	def has_lite_residual_module(net):
		for m in net.modules():
			if isinstance(m, LiteResidualModule):
				return True
		return False

	@property
	def in_channels(self):
		return self.lite_residual_config['in_channels']

	@property
	def out_channels(self):
		return self.lite_residual_config['out_channels']


class ReducedMBConvLayer(MyModule):

	def __init__(self, in_channels, out_channels,
	             kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False, groups=1):
		super(ReducedMBConvLayer, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.kernel_size = kernel_size
		self.stride = stride
		self.expand_ratio = expand_ratio
		self.mid_channels = mid_channels
		self.act_func = act_func
		self.use_se = use_se
		self.groups = groups

		if self.mid_channels is None:
			feature_dim = round(self.in_channels * self.expand_ratio)
		else:
			feature_dim = self.mid_channels

		pad = get_same_padding(self.kernel_size)
		groups = feature_dim if self.groups is None else min_divisible_value(feature_dim, self.groups)
		self.expand_conv = nn.Sequential(OrderedDict({
			'conv': nn.Conv2d(in_channels, feature_dim, kernel_size, stride, pad, groups=groups, bias=False),
			'bn': nn.BatchNorm2d(feature_dim),
			'act': build_activation(self.act_func, inplace=True),
		}))
		if self.use_se:
			self.expand_conv.add_module('se', SEModule(feature_dim))

		self.reduce_conv = nn.Sequential(OrderedDict({
			'conv': nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False),
			'bn': nn.BatchNorm2d(out_channels),
		}))

	def forward(self, x):
		x = self.expand_conv(x)
		x = self.reduce_conv(x)
		return x

	@property
	def module_str(self):
		if self.mid_channels is None:
			expand_ratio = self.expand_ratio
		else:
			expand_ratio = self.mid_channels // self.in_channels
		layer_str = '%dx%d_ReducedMBConv%.3f_%s' % (
			self.kernel_size, self.kernel_size, expand_ratio, self.act_func.upper())
		if self.use_se:
			layer_str = 'SE_' + layer_str
		layer_str += '_O%d' % self.out_channels
		if self.groups is not None:
			layer_str += '_G%d' % self.groups
		if isinstance(self.reduce_conv.bn, nn.GroupNorm):
			layer_str += '_GN%d' % self.reduce_conv.bn.num_groups
		elif isinstance(self.reduce_conv.bn, nn.BatchNorm2d):
			layer_str += '_BN'

		return layer_str

	@property
	def config(self):
		return {
			'name': ReducedMBConvLayer.__name__,
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'expand_ratio': self.expand_ratio,
			'mid_channels': self.mid_channels,
			'act_func': self.act_func,
			'use_se': self.use_se,
			'groups': self.groups,
		}

	@staticmethod
	def build_from_config(config):
		return ReducedMBConvLayer(**config)
