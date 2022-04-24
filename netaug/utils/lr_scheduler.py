import math
from typing import List

import torch
from torch.optim import Optimizer

__all__ = ["CosineLRwithWarmup"]


class CosineLRwithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_lr: float,
        decay_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            return [
                (base_lr - self.warmup_lr) * self.last_epoch / self.warmup_steps
                + self.warmup_lr
                for base_lr in self.base_lrs
            ]
        else:
            current_steps = self.last_epoch - self.warmup_steps
            return [
                0.5
                * base_lr
                * (1 + math.cos(math.pi * current_steps / self.decay_steps))
                for base_lr in self.base_lrs
            ]
