import argparse
import copy
import os
import time
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchpack import distributed as dist
from tqdm import tqdm

from models.netaug import reset_bn
from setup import augemnt_model, build_data_loader, build_model
from utils import (
    AverageMeter,
    CosineLRwithWarmup,
    CrossEntropyWithLabelSmooth,
    DistributedMetric,
    accuracy,
    inference_macs,
    init_modules,
    load_state_dict,
    load_state_dict_from_file,
    parse_unknown_args,
    partial_update_config,
    remove_bn,
    trainable_param_num,
)

parser = argparse.ArgumentParser()

parser.add_argument("config", metavar="FILE", help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument(
    "--gpu", type=str, default=None
)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")

# initialization
parser.add_argument("--init_from", type=str, default=None)


def eval(model: nn.Module, data_loader_dict: Dict, _reset_bn=False) -> Dict:
    if _reset_bn:
        reset_bn(
            model,
            data_loader_dict["sub_train"],
            sync=True,
        )

    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(data_loader_dict["val"]),
            desc="Eval",
            disable=not dist.is_master(),
        ) as t:
            for images, labels in data_loader_dict["val"]:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                loss = test_criterion(output, labels)
                val_loss.update(loss, images.shape[0])
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                val_top5.update(acc5[0], images.shape[0])
                val_top1.update(acc1[0], images.shape[0])

                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "top1": val_top1.avg.item(),
                        "top5": val_top5.avg.item(),
                        "#samples": val_top1.count.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                    }
                )
                t.update()

    val_results = {
        "val_top1": val_top1.avg.item(),
        "val_top5": val_top5.avg.item(),
        "val_loss": val_loss.avg.item(),
    }
    return val_results


def train_one_epoch(
    model: nn.Module,
    data_provider: Dict,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,
    exp_config: Dict,
    netaug_mode: Optional[str] = None,
) -> Dict:
    train_loss = DistributedMetric()
    train_top1 = DistributedMetric()

    model.train()
    data_provider["train"].sampler.set_epoch(epoch)

    data_time = AverageMeter()
    with tqdm(
        total=len(data_provider["train"]),
        desc="Train Epoch #{}".format(epoch + 1),
        disable=not dist.is_master(),
    ) as t:
        end = time.time()
        for _, (images, labels) in enumerate(data_provider["train"]):
            data_time.update(time.time() - end)
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            if netaug_mode is not None:
                # base
                model.module.set_active(mode="min")
                with model.no_sync():
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    top1 = accuracy(output, labels, topk=(1,))[0][0]
                # aug
                model.module.set_active(
                    mode="random" if netaug_mode == "default" else netaug_mode,
                    sync=exp_config["netaug"]["sync"],
                    generator=exp_config["generator"],
                )
                output = model(images)
                aug_loss = criterion(output, labels)
                aug_loss.backward()
            else:
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                top1 = accuracy(output, labels, topk=(1,))[0][0]
            optimizer.step()
            lr_scheduler.step()

            train_loss.update(loss, images.shape[0])
            train_top1.update(top1, images.shape[0])

            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "top1": train_top1.avg.item(),
                    "batch_size": images.shape[0],
                    "img_size": images.shape[2],
                    "lr": optimizer.param_groups[0]["lr"],
                    "data_time": data_time.avg,
                    "netaug": netaug_mode,
                }
            )
            t.update()

            end = time.time()
    return {
        "train_top1": train_top1.avg.item(),
        "train_loss": train_loss.avg.item(),
    }


def train(
    model: nn.Module,
    data_provider: Dict,
    exp_config: Dict,
    path: str,
    resume=False,
    use_netaug=False,
):
    # build optimizer
    params_without_wd = []
    params_with_wd = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                params_with_wd.append(param)
    net_params = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": exp_config["run_config"]["weight_decay"],
        },
    ]
    optimizer = torch.optim.SGD(
        net_params,
        lr=exp_config["run_config"]["base_lr"] * dist.size(),
        momentum=0.9,
        nesterov=True,
    )
    # build lr scheduler
    lr_scheduler = CosineLRwithWarmup(
        optimizer,
        exp_config["run_config"]["warmup_epochs"] * len(data_provider["train"]),
        exp_config["run_config"]["base_lr"],
        exp_config["run_config"]["n_epochs"] * len(data_provider["train"]),
    )
    # train criterion
    train_criterion = CrossEntropyWithLabelSmooth(
        smooth_ratio=exp_config["run_config"]["label_smoothing"]
    )
    # init
    best_val = 0.0
    start_epoch = 0
    checkpoint_path = os.path.join(path, "checkpoint")
    log_path = os.path.join(path, "logs")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logs_writer = open(os.path.join(log_path, "exp.log"), "a")

    if resume and os.path.isfile(os.path.join(checkpoint_path, "checkpoint.pt")):
        checkpoint = torch.load(
            os.path.join(checkpoint_path, "checkpoint.pt"), map_location="cpu"
        )
        model.module.load_state_dict(checkpoint["state_dict"])
        if "best_val" in checkpoint:
            best_val = checkpoint["best_val"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # start training
    for epoch in range(
        start_epoch,
        exp_config["run_config"]["n_epochs"]
        + exp_config["run_config"]["warmup_epochs"],
    ):
        remaining_epochs = (
            exp_config["run_config"]["n_epochs"]
            + exp_config["run_config"]["warmup_epochs"]
            - epoch
        )

        netaug_mode = None
        if use_netaug:
            netaug_mode = "default"
            if remaining_epochs <= exp_config["netaug"]["stop_aug_w_epoch"]:
                netaug_mode = "min_w"
            elif remaining_epochs <= exp_config["netaug"]["stop_aug_e_epoch"]:
                netaug_mode = "min_e"

            if remaining_epochs <= exp_config["netaug"]["stop_netaug_epoch"]:
                netaug_mode = None
            # sort channel
            if exp_config["netaug"]["sort_channel"] and netaug_mode == "default":
                model.module.sort_channels()
                print("sort channels")
            if netaug_mode is None:
                model.module.set_active(mode="min")
        train_info_dict = train_one_epoch(
            model,
            data_provider,
            epoch,
            optimizer,
            train_criterion,
            lr_scheduler,
            exp_config,
            netaug_mode,
        )
        if use_netaug:
            model.module.set_active(mode="min")
        val_info_dict = eval(model, data_provider, use_netaug)
        is_best = val_info_dict["val_top1"] > best_val
        best_val = max(best_val, val_info_dict["val_top1"])
        # log
        epoch_log = f"[{epoch + 1 - exp_config['run_config']['warmup_epochs']}/{exp_config['run_config']['n_epochs']}]"
        epoch_log += f"\tval_top1={val_info_dict['val_top1']:.2f} ({best_val:.2f})"
        epoch_log += f"\ttrain_top1={train_info_dict['train_top1']:.2f}\tlr={optimizer.param_groups[0]['lr']:.2E}"
        if dist.is_master():
            logs_writer.write(epoch_log + "\n")
            logs_writer.flush()

        # save checkpoint
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        if dist.is_master():
            torch.save(
                checkpoint,
                os.path.join(checkpoint_path, "checkpoint.pt"),
                _use_new_zipfile_serialization=False,
            )
            if is_best:
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_path, "best.pt"),
                    _use_new_zipfile_serialization=False,
                )

    # export if use_netaug
    if use_netaug:
        checkpoint = load_state_dict_from_file(os.path.join(checkpoint_path, "best.pt"))
        model.module.load_state_dict(checkpoint)
        model.eval()
        model.module.set_active(mode="min")
        with torch.no_grad():
            model.module(
                torch.zeros(
                    1,
                    3,
                    exp_config["data_provider"]["image_size"],
                    exp_config["data_provider"]["image_size"],
                ).cuda()
            )
        export_model = model.module.export()
        if dist.is_master():
            torch.save(
                {"state_dict": export_model.state_dict()},
                os.path.join(checkpoint_path, "target.pt"),
                _use_new_zipfile_serialization=False,
            )


def main():
    warnings.filterwarnings("ignore")
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # setup gpu and distributed training
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.distributed.is_initialized():
        dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # setup path
    os.makedirs(args.path, exist_ok=True)

    # setup random seed
    if args.resume:
        args.manual_seed = int(time.time())
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)

    # load config
    exp_config = yaml.safe_load(open(args.config, "r"))
    partial_update_config(exp_config, opt)
    # save config to run directory
    yaml.dump(
        exp_config, open(os.path.join(args.path, "config.yaml"), "w"), sort_keys=False
    )

    # build data_loader
    image_size = exp_config["data_provider"]["image_size"]
    data_provider, n_classes = build_data_loader(
        exp_config["data_provider"]["dataset"],
        image_size,
        exp_config["data_provider"]["base_batch_size"],
        exp_config["data_provider"]["n_worker"],
        exp_config["data_provider"]["data_path"],
        dist.size(),
        dist.rank(),
    )

    # build model
    model = build_model(
        exp_config["model"]["name"],
        n_classes,
        exp_config["model"]["dropout_rate"],
    )
    print(model)

    # netaug
    if exp_config.get("netaug", None) is not None:
        use_netaug = True
        model = augemnt_model(
            model, exp_config["netaug"], n_classes, exp_config["model"]["dropout_rate"]
        )
        model.set_active(mode="min")
    else:
        use_netaug = False

    # load init
    if args.init_from is not None:
        init = load_state_dict_from_file(args.init_from)
        load_state_dict(model, init, strict=False)
        print("Loaded init from %s" % args.init_from)
    else:
        init_modules(model, init_type=exp_config["run_config"]["init_type"])
        print("Random Init")

    # profile
    profile_model = copy.deepcopy(model)
    # during inference, bn will be fused into conv
    remove_bn(profile_model)
    print(f"Params: {trainable_param_num(profile_model)}M")
    print(
        f"MACs: {inference_macs(profile_model, data_shape=(1, 3, image_size, image_size))}M"
    )

    # train
    exp_config["generator"] = torch.Generator()
    exp_config["generator"].manual_seed(args.manual_seed)
    model = nn.parallel.DistributedDataParallel(
        model.cuda(), device_ids=[dist.local_rank()]
    )
    train(model, data_provider, exp_config, args.path, args.resume, use_netaug)


if __name__ == "__main__":
    main()
