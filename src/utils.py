import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


class EarlyStopping:
    def __init__(self, patience=7, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value):
        score = -metric_value if self.mode == "min" else metric_value

        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        self.min_lr_ratios = []
        for group in optimizer.param_groups:
            ratio = min_lr / max(group["lr"], 1e-12)
            self.min_lr_ratios.append(ratio)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_step = self.last_epoch

        # linear warmup phase
        if curr_step < self.warmup_steps:
            scale = curr_step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # cosine annealing phase
        progress = (curr_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            base_lr * (ratio + (1 - ratio) * cosine)
            for base_lr, ratio in zip(self.base_lrs, self.min_lr_ratios)
        ]


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    return OmegaConf.load(config_path)


def save_checkpoint(state, is_best, checkpoint_dir, filename="last.pt"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = state["epoch"]
    filename = f"checkpoint_epoch_{epoch}.pt"
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    last_path = os.path.join(checkpoint_dir, "last.pt")
    shutil.copyfile(filepath, last_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        shutil.copyfile(filepath, best_path)


def check_dataset(data_dir):
    data_path = Path(data_dir)
    corrupt_files = []

    print(f"Checking images in {data_dir}...")

    for img_path in data_path.glob("**/*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            try:
                with Image.open(img_path) as img:
                    img.verify()

            except Exception as e:
                print(f"CORRUPT: {img_path} | Error: {e}")
                corrupt_files.append(img_path)

    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupted files.")
    else:
        print("Dataset is clean")
