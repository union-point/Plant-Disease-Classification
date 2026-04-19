import timm
import torch
import torch.nn as nn


class PlantDiseaseModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.backbone_name = config.model.backbone

        self.model = timm.create_model(
            self.backbone_name,
            pretrained=config.model.pretrained,
            num_classes=num_classes,
            drop_rate=config.model.dropout,
            drop_path_rate=config.model.drop_path,
        )

        if config.model.freeze_backbone:
            self._freeze_backbone()
        if config.model.freeze_bn:
            self.freeze_bn()

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, "get_classifier"):
            classifier = self.model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True
        else:
            for name, param in self.model.named_parameters():
                if "head" in name or "classifier" in name:
                    param.requires_grad = True

    def freeze_bn(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()

                if module.weight is not None:
                    module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

    def forward(self, x):
        return self.model(x)


def get_param_groups(model, base_lr, head_lr, weight_decay):
    if hasattr(model.model, "get_classifier"):
        head = model.model.get_classifier()
        head_params = list(head.parameters())
        head_param_ids = set(id(p) for p in head_params)
    else:
        # fallback
        head_params = []
        for name, p in model.named_parameters():
            if any(k in name for k in ["head", "classifier"]):
                head_params.append(p)
        head_param_ids = set(id(p) for p in head_params)

    head_params = [p for p in head_params if p.requires_grad]

    backbone_params = [
        p for p in model.parameters() if id(p) not in head_param_ids and p.requires_grad
    ]
    return [
        {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]
