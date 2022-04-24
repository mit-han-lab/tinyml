import torch
import torch.nn.functional as F

__all__ = ["label_smooth", "CrossEntropyWithSoftTarget", "CrossEntropyWithLabelSmooth"]


def label_smooth(
    target: torch.Tensor, n_classes: int, smooth_factor=0.1
) -> torch.Tensor:
    # convert to one-hot
    batch_size = target.shape[0]
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = torch.add(
        soft_target * (1 - smooth_factor), smooth_factor / n_classes
    )
    return soft_target


class CrossEntropyWithSoftTarget:
    @staticmethod
    def get_loss(pred: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.sum(-soft_target * F.log_softmax(pred, dim=-1, _stacklevel=5), 1)
        )

    def __call__(self, pred: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        return self.get_loss(pred, soft_target)


class CrossEntropyWithLabelSmooth:
    def __init__(self, smooth_ratio=0.1):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.smooth_ratio = smooth_ratio

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        soft_target = label_smooth(target, pred.shape[1], self.smooth_ratio)
        return CrossEntropyWithSoftTarget.get_loss(pred, soft_target)
