import copy
import math
import os.path
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.base import (
    MCUNet,
    MobileNetV2,
    MobileNetV3,
    ProxylessNASMobile,
    TinyMobileNetV2,
)
from models.netaug import (
    NetAugMCUNet,
    NetAugMobileNetV2,
    NetAugMobileNetV3,
    NetAugProxylessNASMobile,
    NetAugTinyMobileNetV2,
)

__all__ = ["build_data_loader", "build_model", "augemnt_model"]


def build_data_loader(
    dataset: str,
    image_size: int,
    batch_size: int,
    n_worker: int = 8,
    data_path: Optional[str] = None,
    num_replica: Optional[int] = None,
    rank: Optional[int] = None,
) -> Tuple[Dict, int]:
    # build dataset
    dataset_info_dict = {
        "imagenet21k_winter_p": (
            os.path.expanduser("~/dataset/imagenet21k_winter_p"),
            10450,
        ),
        "imagenet": (os.path.expanduser("~/dataset/imagenet"), 1000),
        "car": (os.path.expanduser("~/dataset/fgvc/stanford_car"), 196),
        "flowers102": (os.path.expanduser("~/dataset/fgvc/flowers102"), 102),
        "food101": (os.path.expanduser("~/dataset/fgvc/food101"), 101),
        "cub200": (os.path.expanduser("~/dataset/fgvc/cub200"), 200),
        "pets": (os.path.expanduser("~/dataset/fgvc/pets"), 37),
    }
    assert dataset in dataset_info_dict, f"Do not support {dataset}"

    data_path = data_path or dataset_info_dict[dataset][0]
    n_classes = dataset_info_dict[dataset][1]

    # build datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, "train"),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, "val"),
        transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    sub_train_dataset = copy.deepcopy(train_dataset)  # used for resetting bn statistics
    if len(sub_train_dataset) > 16000:
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(len(sub_train_dataset), generator=g).tolist()
        rand_indexes = rand_indexes[:16000]
        sub_train_dataset.samples = [
            sub_train_dataset.samples[idx] for idx in rand_indexes
        ]

    # build data loader
    if num_replica is None:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        sub_train_loader = torch.utils.data.DataLoader(
            dataset=sub_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        sub_train_loader = torch.utils.data.DataLoader(
            dataset=sub_train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                sub_train_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=True,
            drop_last=False,
        )

    # prefetch sub_train
    sub_train_loader = [data for data in sub_train_loader]

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader,
        "sub_train": sub_train_loader,
    }

    return data_loader_dict, n_classes


def build_model(
    name: str,
    n_classes=1000,
    dropout_rate=0.0,
    **kwargs,
) -> nn.Module:

    model_dict = {
        "mbv2": MobileNetV2,
        "mbv3": MobileNetV3,
        "mcunet": MCUNet,
        "proxylessnas": ProxylessNASMobile,
        "tinymbv2": TinyMobileNetV2,
    }

    name = name.split("-")
    if len(name) > 1:
        kwargs["width_mult"] = float(name[1])
    name = name[0]

    return model_dict[name](n_classes=n_classes, dropout_rate=dropout_rate, **kwargs)


def augemnt_model(
    base_model: nn.Module, aug_config: Dict, n_classes=1000, dropout_rate=0.0
) -> nn.Module:
    class_mapping: Dict[Type, Type] = {
        MobileNetV2: NetAugMobileNetV2,
        TinyMobileNetV2: NetAugTinyMobileNetV2,
        ProxylessNASMobile: NetAugProxylessNASMobile,
        MCUNet: NetAugMCUNet,
        MobileNetV3: NetAugMobileNetV3,
    }
    return class_mapping[type(base_model)](
        base_model,
        aug_expand_list=aug_config["aug_expand_list"],
        aug_width_mult_list=aug_config["aug_width_mult_list"],
        n_classes=n_classes,
        dropout_rate=dropout_rate,
    )
