import os
from tqdm import tqdm
import json

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import datasets, transforms

from mcunet.model_zoo import build_model
from mcunet.utils import AverageMeter, accuracy, count_net_flops, count_parameters

# Training settings
parser = argparse.ArgumentParser()
# net setting
parser.add_argument('--net_id', type=str, help='net id of the model')
# data loader setting
parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'vww'])
parser.add_argument('--data-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                    help='path to ImageNet validation data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = 'cuda'


def build_val_data_loader(resolution):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {'num_workers': args.workers, 'pin_memory': True}

    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize
        ])
    elif args.dataset == 'vww':
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc='Validate') as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({'loss': val_loss.avg,
                               'top1': val_top1.avg})
                t.update(1)

    return val_top1.avg


def main():
    model, resolution, description = build_model(args.net_id, pretrained=True)
    model = model.to(device)
    model.eval()
    val_loader = build_val_data_loader(resolution)

    # profile model
    total_macs = count_net_flops(model, [1, 3, resolution, resolution])
    total_params = count_parameters(model)
    print(' * FLOPs: {:.4}M, param: {:.4}M'.format(total_macs / 1e6, total_params / 1e6))

    acc = validate(model, val_loader)
    print(' * Accuracy: {:.2f}%'.format(acc))


if __name__ == '__main__':
    main()
