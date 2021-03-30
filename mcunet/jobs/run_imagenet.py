import sys
sys.path.append(".")

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
import json
from utils import DistributedMetric, accuracy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--job', default=None, type=str)
parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
parser.add_argument('--checkpoint-dir', default='./checkpoint',
                    help='checkpoint file format')
# architecture setting
parser.add_argument('-a', '--arch', metavar='ARCH', default='proxyless')
parser.add_argument('--net_config', default=None, type=str)
# data setting
parser.add_argument('--train-dir', default=os.path.expanduser('/dataset/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('/dataset/imagenet/val'),
                    help='path to validation data')
parser.add_argument('--resolution', default=None, type=int)  # will set from model config
# training hyper-params
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--lr_type', type=str, default='cosine')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
# resuming from previous weights
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--load_from', default=None, type=str,
                    help='load from a checkpoint, either for evaluation or fine-tuning')

parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--calibrate', action='store_true', default=False)
# extra techniques (not used for paper results)
parser.add_argument('--mixup-alpha', default=0, type=float, help='The alpha value used in mix up training')
parser.add_argument('--label_smoothing', type=float, default=0)

args = parser.parse_args()

hvd.init()

# Horovod: pin GPU to local rank.
torch.cuda.set_device(hvd.local_rank())
cudnn.benchmark = True

device = 'cuda'
log_writer = None

verbose = 1 if hvd.rank() == 0 else 0

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(args.workers)

# create model
if args.arch == 'proxyless':
    from tinynas.nn.networks import ProxylessNASNets
    with open(args.net_config) as f:
        config = json.load(f)
        args.resolution = config['resolution']
    model = ProxylessNASNets.build_from_config(config)
else:
    raise NotImplementedError
model = model.to(device)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
kwargs = {'num_workers': args.workers, 'pin_memory': True}

train_dataset = datasets.ImageFolder(args.train_dir,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(args.resolution),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize

                                     ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = datasets.ImageFolder(args.val_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(int(args.resolution * 256 / 224)),
                                       transforms.CenterCrop(args.resolution),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                         sampler=val_sampler, **kwargs)

if hvd.rank() == 0:
    from utils import count_net_flops
    total_ops = count_net_flops(model, [1, 3, args.resolution, args.resolution])
    total_params = sum([p.numel() for p in model.parameters()])
    print(' * FLOPs: {:.4}M, param: {:.4}M'.format(total_ops / 1e6, total_params / 1e6))

if args.load_from is not None:
    sd = torch.load(args.load_from, map_location='cpu')
    model.load_state_dict(sd['state_dict'])

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.lr * hvd.size()),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
)

resume_from_epoch = 0

if args.resume:
    ckpt_path = os.path.join(args.checkpoint_dir, args.job, 'ckpt.pth.tar')
    if os.path.exists(ckpt_path):
        if hvd.rank() == 0:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            resume_from_epoch = checkpoint['epoch']
            print(' * Loading from checkpoint @ epoch {}'.format(resume_from_epoch))
    else:
        if hvd.rank() == 0:
            print(' * Checkpoint not found!')

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

if hvd.rank() == 0 and args.calibrate:
    # re-compute the bn statistics
    from utils import set_running_statistics
    set_running_statistics(model, train_loader, distributed=False, maximum_iter=200)
    # print(' TODODODODO')
    # torch.save({'state_dict': model.cpu().state_dict()},
    #            'assets/pt_ckpt/mcunet-10fps_imagenet.pth')
    # model.to(device)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = DistributedMetric('train_loss')
    train_top1 = DistributedMetric('train_top1')
    train_top5 = DistributedMetric('train_top5')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            this_lr = adjust_learning_rate(epoch, batch_idx)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if args.mixup_alpha > 0:  # mix up training
                from lib.mix_up import mixup_data, mixup_criterion
                inputs, targets_a, targets_b, lam = mixup_data(data, target, args.mixup_alpha)
                output = model(inputs)
                loss = mixup_criterion(nn.CrossEntropyLoss(), output, targets_a, targets_b, lam)
            elif args.label_smoothing > 0:
                output = model(data)
                from utils.pytorch_utils import cross_entropy_with_label_smoothing
                loss = cross_entropy_with_label_smoothing(output, target, args.label_smoothing)
            else:
                output = model(data)
                loss = F.cross_entropy(output, target)

            top1, top5 = accuracy(output, target, topk=(1, 5))

            train_top1.update(top1)
            train_top5.update(top5)
            train_loss.update(loss)
            loss.backward()
            optimizer.step()
            t.set_postfix({'lr': this_lr,
                           'loss': train_loss.avg.item(),
                           'top1': train_top1.avg.item(),
                           'top5': train_top5.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/top1', train_top1.avg, epoch)
        log_writer.add_scalar('train/top5', train_top5.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = DistributedMetric('val_loss')
    val_top1 = DistributedMetric('val_top1')
    val_top5 = DistributedMetric('val_top5')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data_target in val_loader:
                data, target = data_target
                data, target = data.to(device), target.to(device)

                output = model(data)
                val_loss.update(F.cross_entropy(output, target))
                top1, top5 = accuracy(output, target, topk=(1, 5))
                val_top1.update(top1)
                val_top5.update(top5)
                t.set_postfix({'loss': val_loss.avg.item(),
                               'top1': val_top1.avg.item(),
                               'top5': val_top5.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/top1', val_top1.avg, epoch)
        log_writer.add_scalar('val/top5', val_top5.avg, epoch)
    return val_top1.avg.item()


# Horovod: using `lr = lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = lr` ---> `lr = lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    train_loader_len = len(train_loader)

    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / train_loader_len
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    else:
        if args.lr_type == 'step':
            if epoch < 10:
                lr_adj = 1.
            elif epoch < 20:
                lr_adj = 1e-1
            else:
                lr_adj = 1e-2
        elif args.lr_type == 'cosine':
            epoch += float(batch_idx + 1) / train_loader_len
            lr_adj = 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        elif args.lr_type == 'exp':
            epoch += float(batch_idx + 1) / train_loader_len
            lr_adj = 0.95 ** epoch
        elif args.lr_type == 'linear':
            epoch += float(batch_idx + 1) / train_loader_len
            lr_adj = 1 - epoch / args.epochs
        elif args.lr_type == 'fixed':
            lr_adj = 1.
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * hvd.size() * lr_adj
    return args.lr * hvd.size() * lr_adj


def save_checkpoint(epoch, is_best):
    checkpoint_path = os.path.join(args.checkpoint_dir, args.job, 'ckpt.pth.tar')
    if hvd.rank() == 0:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(state, checkpoint_path)
        if is_best:
            import shutil
            shutil.copyfile(checkpoint_path, checkpoint_path.replace('.pth.tar', '.best.pth.tar'))


if args.evaluate:
    acc = validate(0)
    if hvd.rank() == 0:
        print(' * Accuracy: {:.2f}%'.format(acc))
    exit()

# set up the directories
assert args.job is not None
if hvd.rank() == 0:
    print('#' * 20, args.job)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, args.job), exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    log_writer = SummaryWriter(os.path.join(args.log_dir, args.job))

best_acc = 0.

for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    acc = validate(epoch)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint(epoch, is_best)
    if is_best and verbose:
        print(' * best acc:', best_acc)