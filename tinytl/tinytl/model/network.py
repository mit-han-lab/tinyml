from ofa.utils.layers import ResidualBlock
from ofa.imagenet_classification.networks import ProxylessNASNets
from .modules import my_set_layer_from_config

__all__ = ['build_residual_block_from_config', 'build_network_from_config']


def build_residual_block_from_config(config):
	conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
	conv = my_set_layer_from_config(conv_config)
	shortcut = my_set_layer_from_config(config['shortcut'])
	return ResidualBlock(conv, shortcut)


def build_network_from_config(config):
	first_conv = my_set_layer_from_config(config['first_conv'])
	feature_mix_layer = my_set_layer_from_config(config['feature_mix_layer'])
	classifier = my_set_layer_from_config(config['classifier'])

	blocks = []
	for block_config in config['blocks']:
		blocks.append(build_residual_block_from_config(block_config))

	net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
	if 'bn' in config:
		net.set_bn_param(**config['bn'])
	else:
		net.set_bn_param(momentum=0.1, eps=1e-3)

	return net
