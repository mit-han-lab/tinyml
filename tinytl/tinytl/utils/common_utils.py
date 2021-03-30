import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.cluster import KMeans

__all__ = [
    'module_require_grad', 'set_module_grad_status', 'enable_bn_update', 'enable_bias_update',
    'weight_quantization',
]


def module_require_grad(module):
    return module.parameters().__next__().requires_grad


def set_module_grad_status(module, flag=False):
    if isinstance(module, list):
        for m in module:
            set_module_grad_status(m, flag)
    else:
        for p in module.parameters():
            p.requires_grad = flag


def enable_bn_update(model):
    for m in model.modules():
        if type(m) in [nn.BatchNorm2d, nn.GroupNorm] and m.weight is not None:
            set_module_grad_status(m, True)


def enable_bias_update(model):
    for m in model.modules():
        for name, param in m.named_parameters():
            if name == 'bias':
                param.requires_grad = True


def k_means_cpu(weight, n_clusters, init='k-means++', max_iter=50):
    # flatten the weight for computing k-means
    org_shape = weight.shape
    weight = weight.reshape(-1, 1)  # single feature
    if n_clusters > weight.size:
        n_clusters = weight.size

    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=max_iter, n_jobs=20)
    k_means.fit(weight)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    labels = labels.reshape(org_shape)
    return torch.from_numpy(centroids).view(1, -1), torch.from_numpy(labels).int()


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight


def quantization(layer, bits=8, max_iter=50):
    w = layer.weight.data
    centroids, labels = k_means_cpu(w.cpu().numpy(), 2 ** bits, max_iter=max_iter)
    w_q = reconstruct_weight_from_k_means_result(centroids, labels)
    layer.weight.data = w_q.float()


def weight_quantization(model, bits=8, max_iter=50):
    if bits is None:
        return
    to_quantize_modules = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if not m.weight.requires_grad:
                to_quantize_modules.append(m)

    with tqdm(total=len(to_quantize_modules),
              desc='%d-bits quantization start' % bits) as t:
        for m in to_quantize_modules:
            quantization(m, bits, max_iter)
            t.update()
