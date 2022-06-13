import math
import copy
import time
import torch
import torch.nn as nn

__all__ = [
    'mix_images', 'mix_labels',
    'label_smooth', 'cross_entropy_loss_with_soft_target', 'cross_entropy_with_label_smoothing',
    'clean_num_batch_tracked', 'rm_bn_from_net',
    'get_net_device', 'count_parameters', 'count_net_flops', 'count_peak_activation_size',
    'measure_net_latency', 'get_net_info',
    'build_optimizer', 'calc_learning_rate',
]

""" Mixup """


def mix_images(images, lam):
    flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
    return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
    onehot_target = label_smooth(target, n_classes, label_smoothing)
    flipped_target = torch.flip(onehot_target, dims=[0])
    return lam * onehot_target + (1 - lam) * flipped_target


""" Label smooth """


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)


""" BN related """


def clean_num_batch_tracked(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x  # exclude from computation
            del m.weight  # exclude model size
            del m.bias
            del m.running_mean
            del m.running_var


""" Network profiling """


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
    from tinynas.nn.networks import MobileInvertedResidualBlock

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
        from tinynas.nn.modules import ZeroLayer
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
                count_conv_mem(m.mobile_inverted_conv.point_linear) # + residual_size,
            ])

    if isinstance(net, nn.DataParallel):
        net = net.module
    net = copy.deepcopy(net)

    from tinynas.nn.networks import ProxylessNASNets
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


def measure_net_latency(net, l_type='gpu8', fast=True, input_shape=(3, 224, 224), clean=False):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if 'gpu' in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == 'cpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device('cpu'):
            if not clean:
                print('move net to cpu for measuring cpu latency')
            net = copy.deepcopy(net).cpu()
    elif l_type == 'gpu':
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    images = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {'warmup': [], 'sample': []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(images)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency['warmup'].append(used_time)
            if not clean:
                print('Warmup %d: %.3f' % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(images)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency['sample'].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=True):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info['params'] = count_parameters(net) / 1e6

    # flops
    net_info['flops'] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split('#')
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(net, l_type, fast=False, input_shape=input_shape)
        net_info['%s latency' % l_type] = {
            'val': latency,
            'hist': measured_latency
        }

    if print_info:
        print(net)
        print('Total training params: %.2fM' % (net_info['params']))
        print('Total FLOPs: %.2fM' % (net_info['flops']))
        for l_type in latency_types:
            print('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))

    return net_info


""" optimizer """


def build_optimizer(net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {'params': net_params[0], 'weight_decay': weight_decay},
            {'params': net_params[1], 'weight_decay': 0},
        ]
    else:
        net_params = [{'params': net_params, 'weight_decay': weight_decay}]

    if opt_type == 'sgd':
        opt_param = {} if opt_param is None else opt_param
        momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
        optimizer = torch.optim.SGD(net_params, init_lr, momentum=momentum, nesterov=nesterov)
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(net_params, init_lr)
    else:
        raise NotImplementedError
    return optimizer


""" learning rate schedule """


def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr
