import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        inputs: logits [B, C]
        targets: labels [B] or soft mixup labels [B, C]
        """
        if targets.ndim == inputs.ndim:
            # targets are soft labels from MixUp/CutMix
            ce_loss = F.cross_entropy(
                inputs, targets, reduction="none", label_smoothing=self.label_smoothing
            )
            # for focal weighting when using mixup, pt is e^(-ce_loss)
            pt = torch.exp(-ce_loss)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, reduction="none", label_smoothing=self.label_smoothing
            )
            pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_criterion(config):
    if config.loss.name == "focal":
        return FocalLoss(
            gamma=config.loss.gamma,
            alpha=config.loss.alpha,
            label_smoothing=config.loss.label_smoothing,
        )
    else:
        if config.augmentation.prob > 0:
            return SoftTargetCrossEntropy()
        return nn.CrossEntropyLoss(label_smoothing=config.loss.label_smoothing)
