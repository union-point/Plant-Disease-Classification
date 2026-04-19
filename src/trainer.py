import torch
import wandb
from omegaconf import OmegaConf
from timm.utils import ModelEmaV2
from torch import nn
from torch.amp import GradScaler, autocast
from torchvision.transforms import v2
from tqdm import tqdm

from .metrics import MetricTracker
from .utils import EarlyStopping, save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience, mode="max"
        )

        self.scaler = GradScaler(device.type, enabled=config.training.mixed_precision)

        self.use_ema = (
            getattr(config.training, "ema", None) and config.training.ema.enabled
        )
        if self.use_ema:
            ema_decay = getattr(config.training.ema, "decay", 0.9999)
            self.model_ema = ModelEmaV2(self.model, decay=ema_decay, device=device)
        else:
            self.model_ema = None

        self.num_classes = config.model.num_classes

        self.use_mixup = False
        if config.augmentation.prob > 0:
            self.use_mixup = True
            cutmix = v2.CutMix(
                alpha=config.augmentation.cutmix_alpha, num_classes=self.num_classes
            )
            mixup = v2.MixUp(
                alpha=config.augmentation.mixup_alpha, num_classes=self.num_classes
            )
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        self.train_metrics = MetricTracker(num_classes=self.num_classes, device=device)
        self.val_metrics = MetricTracker(num_classes=self.num_classes, device=device)
        if self.use_ema:
            self.val_ema_metrics = MetricTracker(
                num_classes=self.num_classes, device=device
            )

    def train_one_epoch(self, epoch):
        self.model.train()
        if self.config.model.freeze_bn:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            is_mixed = False

            # apply MixUp or CutMix
            if self.use_mixup and torch.rand(1).item() < self.config.augmentation.prob:
                images, targets = self.cutmix_or_mixup(images, targets)
                is_mixed = True
            if targets.ndim == 1:
                targets = torch.nn.functional.one_hot(
                    targets, num_classes=self.num_classes
                ).float()
            with autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                # gradient accumulation normalizer
                loss = loss / self.config.training.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.clip_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.config.scheduler.name == "cosine_warmup":
                    self.scheduler.step()

                if self.use_ema:
                    self.model_ema.update(self.model)

            batch_loss = loss.item() * self.config.training.gradient_accumulation_steps
            self.train_metrics.update(
                outputs.detach(),
                targets.detach(),
                loss=batch_loss,
                skip_metrics=is_mixed,
            )

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            if self.config.logging.use_wandb:
                wandb.log({"train/batch_loss": batch_loss})

        if (batch_idx + 1) % self.config.training.gradient_accumulation_steps != 0:
            if self.config.training.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.clip_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.config.scheduler.name == "cosine_warmup":
                self.scheduler.step()

            if self.use_ema:
                self.model_ema.update(self.model)

        metrics = self.train_metrics.compute()

        # Step schedulers that step per epoch
        if self.config.scheduler.name == "step":
            self.scheduler.step()
        elif self.config.scheduler.name == "cosine":
            self.scheduler.step()

        return metrics

    def validate(self, epoch):
        self.model.eval()
        self.val_metrics.reset()

        if self.use_ema:
            self.model_ema.module.eval()
            self.val_ema_metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)

                if targets.ndim == 1:
                    targets = torch.nn.functional.one_hot(
                        targets, num_classes=self.num_classes
                    ).float()

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.training.mixed_precision,
                ):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                    if self.use_ema:
                        ema_outputs = self.model_ema.module(images)
                        ema_loss = self.criterion(ema_outputs, targets)

                self.val_metrics.update(
                    outputs.detach(), targets.detach(), loss=loss.detach()
                )
                if self.use_ema:
                    self.val_ema_metrics.update(
                        ema_outputs.detach(), targets.detach(), loss=ema_loss.detach()
                    )
                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "ema_loss": f"{ema_loss.item():.4f}",
                        }
                    )
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = {"current": self.val_metrics.compute()}
        if self.use_ema:
            metrics["ema"] = self.val_ema_metrics.compute()

        primary_map = metrics[self.config.training.ema.eval_mode]["mAP"]

        if self.config.scheduler.name == "plateau":
            self.scheduler.step(primary_map)

        return metrics

    def fit(self, start_epoch=1):
        best_map = 0.0

        for epoch in range(start_epoch, self.config.training.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            lrs = [pg["lr"] for pg in self.optimizer.param_groups]

            log_dict = {
                "train/loss": train_metrics["loss"],
                "train/mAP": train_metrics["mAP"],
                "train/accuracy": train_metrics["accuracy"],
                "lr/backbone": lrs[0],
                "lr/head": lrs[1],
                "epoch": epoch,
            }

            if self.use_ema:
                log_dict.update(
                    {
                        "val/loss": val_metrics["current"]["loss"],
                        "val/mAP": val_metrics["current"]["mAP"],
                        "val/accuracy": val_metrics["current"]["accuracy"],
                        "val/ema_loss": val_metrics["ema"]["loss"],
                        "val/ema_mAP": val_metrics["ema"]["mAP"],
                        "val/ema_accuracy": val_metrics["ema"]["accuracy"],
                    }
                )
            else:
                log_dict.update(
                    {
                        "val/loss": val_metrics["current"]["loss"],
                        "val/mAP": val_metrics["current"]["mAP"],
                        "val/accuracy": val_metrics["current"]["accuracy"],
                    }
                )

            if self.config.logging.use_wandb:
                wandb.log(log_dict)

            print(f"\nEpoch {epoch} Summary:")
            print(f"LR: Backbone: {lrs[0]:.2e} | Head: {lrs[1]:.2e}")
            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, mAP: {train_metrics['mAP']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            )
            if self.use_ema:
                print(
                    f"Val (Current) - Loss: {val_metrics['current']['loss']:.4f}, mAP: {val_metrics['current']['mAP']:.4f}, Acc: {val_metrics['current']['accuracy']:.4f}"
                )
                print(
                    f"Val (EMA)     - Loss: {val_metrics['ema']['loss']:.4f}, mAP: {val_metrics['ema']['mAP']:.4f}, Acc: {val_metrics['ema']['accuracy']:.4f}"
                )
            else:
                print(
                    f"Val   - Loss: {val_metrics['current']['loss']:.4f}, mAP: {val_metrics['current']['mAP']:.4f}, Acc: {val_metrics['current']['accuracy']:.4f}"
                )

            primary_map = val_metrics[self.config.training.ema.eval_mode]["mAP"]
            is_best = self.early_stopping(primary_map)

            if is_best:
                best_map = primary_map
                print(f"Epoch {epoch} is the new best model. mAP: {best_map:.4f}")

            # Checkpointing
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "state_dict_ema": self.model_ema.module.state_dict()
                if self.use_ema
                else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "scaler": self.scaler.state_dict(),
                "early_stopping": {
                    "best_score": self.early_stopping.best_score,
                    "counter": self.early_stopping.counter,
                    "early_stop": self.early_stopping.early_stop,
                },
                "rng_states": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                "val_mAP": primary_map,
                "config": OmegaConf.to_yaml(self.config),
                "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            }
            save_checkpoint(state, is_best, self.config.logging.checkpoint_dir)

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("Training complete!")
