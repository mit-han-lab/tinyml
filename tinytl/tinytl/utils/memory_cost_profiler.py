import copy
import torch
import torch.nn as nn
from ofa.utils import Hswish, Hsigmoid, MyConv2d

from ofa.utils.layers import ResidualBlock
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.mobilenet import InvertedResidual

__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
	frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

	trainable_param_size = 0
	frozen_param_size = 0
	for p in net.parameters():
		if p.requires_grad:
			trainable_param_size += trainable_param_bits / 8 * p.numel()
		else:
			frozen_param_size += frozen_param_bits / 8 * p.numel()
	model_size = trainable_param_size + frozen_param_size
	if print_log:
		print('Total: %d' % model_size,
		      '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
		      '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
	# Byte
	return model_size


def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32):
	act_byte = activation_bits / 8
	model = copy.deepcopy(net)

	# noinspection PyArgumentList
	def count_convNd(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

	# noinspection PyArgumentList
	def count_linear(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_bn(m, x, _):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_relu(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_smooth_act(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	def add_hooks(m_):
		if len(list(m_.children())) > 0:
			return

		m_.register_buffer('grad_activations', torch.zeros(1))
		m_.register_buffer('tmp_activations', torch.zeros(1))

		if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d, MyConv2d]:
			fn = count_convNd
		elif type(m_) in [nn.Linear]:
			fn = count_linear
		elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
			fn = count_bn
		elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
			fn = count_relu
		elif type(m_) in [nn.Sigmoid, nn.Tanh, Hswish, Hsigmoid]:
			fn = count_smooth_act
		else:
			fn = None

		if fn is not None:
			_handler = m_.register_forward_hook(fn)

	model.eval()
	model.apply(add_hooks)

	x = torch.zeros(input_size).to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x)

	memory_info_dict = {
		'peak_activation_size': torch.zeros(1),
		'grad_activation_size': torch.zeros(1),
		'residual_size': torch.zeros(1),
	}

	for m in model.modules():
		if len(list(m.children())) == 0:
			def new_forward(_module):
				def lambda_forward(_x):
					current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
					                   memory_info_dict['residual_size']
					memory_info_dict['peak_activation_size'] = max(
						current_act_size, memory_info_dict['peak_activation_size']
					)
					memory_info_dict['grad_activation_size'] += _module.grad_activations
					return _module.old_forward(_x)

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		if (isinstance(m, ResidualBlock) and m.shortcut is not None) or \
				(isinstance(m, InvertedResidual) and m.use_res_connect) or \
				type(m) in [BasicBlock, Bottleneck]:
			def new_forward(_module):
				def lambda_forward(_x):
					memory_info_dict['residual_size'] = _x.numel() * act_byte
					result = _module.old_forward(_x)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

	with torch.no_grad():
		model(x)

	return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
	param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
	activation_size, _ = count_activation_size(net, input_size, require_backward, activation_bits)

	memory_cost = activation_size * batch_size + param_size
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}
