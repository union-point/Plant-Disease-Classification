import argparse
import math
import os

import torch
import wandb
from omegaconf import OmegaConf
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.dataset import get_dataloaders
from src.loss import get_criterion
from src.models import PlantDiseaseModel, get_param_groups
from src.trainer import Trainer
from src.utils import CosineAnnealingWarmupLR, load_config, set_seed


def build_optimizer(model, config):
    layer_decay = getattr(config.optimizer, "layer_decay", 1.0)
    param_groups = get_param_groups(
        model,
        base_lr=config.optimizer.backbone_lr,
        head_lr=config.optimizer.head_lr,
        weight_decay=config.optimizer.weight_decay,
    )

    if config.optimizer.name.lower() == "adamw":
        if layer_decay == 1:
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = create_optimizer_v2(
                model,
                opt="adamw",
                lr=config.optimizer.head_lr,
                layer_decay=layer_decay,
                weight_decay=config.optimizer.weight_decay,
            )
    else:
        optimizer = torch.optim.Adam(param_groups)

    return optimizer


def build_scheduler(optimizer, config, len_loader):
    if config.scheduler.name.lower() == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=config.scheduler.min_lr
        )
    elif config.scheduler.name.lower() == "step":
        return StepLR(optimizer, step_size=3, gamma=0.1)
    elif config.scheduler.name.lower() == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=3,
            min_lr=config.scheduler.min_lr,
        )
    elif config.scheduler.name.lower() == "cosine_warmup":
        return CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=config.scheduler.warmup_epochs
            * len_loader
            / config.training.gradient_accumulation_steps,
            total_steps=config.training.epochs
            * len_loader
            / config.training.gradient_accumulation_steps,
            min_lr=config.scheduler.min_lr,
        )
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Plant Disease Classification Baseline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--init_weights", type=str, default=None, help="Path to weights for warm start"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config.seed, deterministic=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment: Using device {device}")

    train_loader, val_loader, num_classes = get_dataloaders(config)

    if num_classes == 0:
        print(
            "WARNING: No data found. Make sure your datasets are correctly structured."
        )
        # Fallback to prevent immediate crash if no data is present yet
        num_classes = 1

    config.model.num_classes = num_classes

    model = PlantDiseaseModel(config, num_classes=num_classes)
    model.to(device)

    if args.init_weights and os.path.exists(args.init_weights):
        print(f"Warm starting from weights: {args.init_weights}")
        checkpoint = torch.load(args.init_weights, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)

    optimizer = build_optimizer(model, config)
    criterion = get_criterion(config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))

    # resume Logic
    start_epoch = 1
    checkpoint = None
    run_id = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming experiment from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1

        if "rng_states" in checkpoint:
            torch.set_rng_state(checkpoint["rng_states"]["torch"].cpu())
            if device.type == "cuda" and checkpoint["rng_states"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in checkpoint["rng_states"]["cuda"]]
                )

        if config.logging.use_wandb:
            run_id = checkpoint.get("wandb_run_id")

        if start_epoch > config.training.epochs:
            print(
                f"Requested to resume at epoch {start_epoch}, but total epochs is {config.training.epochs}. Exiting."
            )
            return

    # Wandb tracking
    if config.logging.use_wandb:
        wandb_config = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project=config.logging.project_name,
            name=config.experiment_name,
            config=wandb_config,
            id=run_id,  # Use the loaded ID (or None if brand new)
            resume="allow",
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    if checkpoint is not None:
        if trainer.use_ema and checkpoint.get("state_dict_ema"):
            trainer.model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

    if args.resume and os.path.exists(args.resume):
        if checkpoint["scaler"]:
            trainer.scaler.load_state_dict(checkpoint["scaler"])

        if checkpoint["early_stopping"]:
            trainer.early_stopping.best_score = checkpoint["early_stopping"][
                "best_score"
            ]
            trainer.early_stopping.counter = checkpoint["early_stopping"]["counter"]
            trainer.early_stopping.early_stop = checkpoint["early_stopping"][
                "early_stop"
            ]

    trainer.fit()


if __name__ == "__main__":
    main()
