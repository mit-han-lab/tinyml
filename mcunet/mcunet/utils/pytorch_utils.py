import copy
import torch
import torch.nn as nn

__all__ = [
    'rm_bn_from_net', 'get_net_device', 'count_parameters', 'count_net_flops',
    'count_peak_activation_size',
]

""" Network profiling """


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x  # exclude from computation
            del m.weight  # exclude model size
            del m.bias
            del m.running_mean
            del m.running_var

def rm_bn(module):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module_output = nn.Identity()

    for name, child in module.named_children():
        module_output.add_module(name, rm_bn(child))
    del module
    return module_output
    


def get_net_device(net):
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(model, data_shape):
    from torchprofile import profile_macs
    model = copy.deepcopy(model)
    rm_bn_from_net(model)  # remove bn since it is eventually fused
    total_macs = profile_macs(model, torch.randn(*data_shape).to(get_net_device(model)))
    del model
    return total_macs


def count_peak_activation_size(net, data_shape=(1, 3, 224, 224)):
    from ..tinynas.nn.networks import MobileInvertedResidualBlock

    def record_in_out_size(m, x, y):
        x = x[0]
        m.input_size = torch.Tensor([x.numel()])
        m.output_size = torch.Tensor([y.numel()])

    def add_io_hooks(m_):
        m_type = type(m_)
        if m_type in [nn.Conv2d, nn.Linear, MobileInvertedResidualBlock]:
            m_.register_buffer('input_size', torch.zeros(1))
            m_.register_buffer('output_size', torch.zeros(1))
            m_.register_forward_hook(record_in_out_size)

    def count_conv_mem(m):
        # we assume we only need to store input and output, the weights are partially loaded for computation
        if m is None:
            return 0
        if hasattr(m, 'conv'):
            m = m.conv
        elif hasattr(m, 'linear'):
            m = m.linear
        assert isinstance(m, (nn.Conv2d, nn.Linear))
        return m.input_size.item() + m.output_size.item()

    def count_block(m):
        from ..tinynas.nn.modules import ZeroLayer
        assert isinstance(m, MobileInvertedResidualBlock)

        if m.mobile_inverted_conv is None or isinstance(m.mobile_inverted_conv, ZeroLayer):  # just an identical mapping
            return 0
        elif m.shortcut is None or isinstance(m.shortcut, ZeroLayer):  # no residual connection, just convs
            return max([
                count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                count_conv_mem(m.mobile_inverted_conv.depth_conv),
                count_conv_mem(m.mobile_inverted_conv.point_linear),
            ])
        else:  # convs and residual
            residual_size = m.mobile_inverted_conv.inverted_bottleneck.conv.input_size.item()
            # consider residual size for later layers
            return max([
                count_conv_mem(m.mobile_inverted_conv.inverted_bottleneck),
                count_conv_mem(m.mobile_inverted_conv.depth_conv) + residual_size,
                # TODO: can we omit the residual here? reuse the output?
                count_conv_mem(m.mobile_inverted_conv.point_linear)  # + residual_size,
            ])

    if isinstance(net, nn.DataParallel):
        net = net.module
    net = copy.deepcopy(net)

    from ..tinynas.nn.networks import ProxylessNASNets
    assert isinstance(net, ProxylessNASNets)

    # record the input and output size
    net.apply(add_io_hooks)
    with torch.no_grad():
        _ = net(torch.randn(*data_shape).to(net.parameters().__next__().device))

    mem_list = [
                   count_conv_mem(net.first_conv),
                   count_conv_mem(net.feature_mix_layer),
                   count_conv_mem(net.classifier)
               ] + [count_block(blk) for blk in net.blocks]

    del net
    return max(mem_list)  # pick the peak mem
