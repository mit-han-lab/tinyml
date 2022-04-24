import argparse
import os

import torch.backends.cudnn
import torch.nn as nn
from torchpack import distributed as dist

from setup import build_data_loader, build_model
from train import eval
from utils import load_state_dict_from_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument(
    "--dataset",
    type=str,
    default="imagenet",
    choices=[
        "imagenet",
        "imagenet21k_winter_p",
        "car",
        "flowers102",
        "food101",
        "cub200",
        "pets",
    ],
)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--image_size", type=int, default=160)

parser.add_argument(
    "--model",
    type=str,
    default="mbv2-0.35",
    choices=[
        "mbv2-0.35",
        "mbv3-0.35",
        "proxylessnas-0.35",
        "mcunet",
        "tinymbv2",
    ],
)

parser.add_argument("--init_from", type=str)
parser.add_argument("--reset_bn", action="store_true")
parser.add_argument("--save_path", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    # setup gpu and distributed training
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # build data loader
    data_loader_dict, n_classes = build_data_loader(
        args.dataset,
        args.image_size,
        args.batch_size,
        args.n_worker,
        args.data_path,
        dist.size(),
        dist.rank(),
    )

    # build model
    model = build_model(args.model, n_classes, 0).cuda()

    # load checkpoint
    checkpoint = load_state_dict_from_file(args.init_from)
    model.load_state_dict(checkpoint)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank()])
    val_results = eval(model, data_loader_dict, args.reset_bn)

    for key, val in val_results.items():
        print(key, ": ", val)

    if args.save_path is not None:
        torch.save(
            model.module.state_dict(),
            args.save_path,
            _use_new_zipfile_serialization=False,
        )
