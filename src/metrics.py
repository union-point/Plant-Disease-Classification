import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassAveragePrecision


class MetricTracker:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

        self.map_metric = MulticlassAveragePrecision(num_classes=num_classes).to(device)
        self.acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)

        self.reset()

    def reset(self):
        self.map_metric.reset()
        self.acc_metric.reset()
        self.loss_sum = 0
        self.count = 0

    def update(self, preds, targets, loss=None, skip_metrics=False):
        """
        preds: logits [B, C]
        targets: [B] or soft labels [B, C]
        skip_metrics: If True, only loss is tracked. Use for MixUp/CutMix batches.
        """
        if targets.ndim > 1:
            hard_targets = targets.argmax(dim=1)
        else:
            hard_targets = targets
        if not skip_metrics:
            self.map_metric.update(preds, hard_targets)
            self.acc_metric.update(preds, hard_targets)

        if loss is not None:
            self.loss_sum += loss * preds.size(0)
            self.count += preds.size(0)

    def compute(self):
        mAP = self.map_metric.compute().item()
        acc = self.acc_metric.compute().item()
        avg_loss = self.loss_sum / max(self.count, 1)
        return {"mAP": mAP, "accuracy": acc, "loss": avg_loss}
