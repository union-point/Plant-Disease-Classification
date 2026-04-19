# Plant Disease Classification

A robust, configurable deep learning pipeline for plant disease classification using PyTorch. This project leverages `timm` for a vast array of pre-trained backbones (e.g., EfficientNetV2, ConvNeXtV2, EVA02) and offers advanced training features such as Exponential Moving Average (EMA) for weights, Layer-wise Learning Rate Decay (LLRD), MixUp/CutMix data augmentation, and Weights & Biases (W&B) integration for experiment tracking.


- **Web Interface:** [](https://huggingface.co/spaces/)
- **REST API Documentation:** [s]()
## Features

- **Extensive Model Support**: Easily swap backbones by changing the config, enabled by integration with `timm`.
- **Advanced Training Techniques**:
  - Model EMA (Exponential Moving Average) to stabilize training and improve generalization.
  - Layer-wise Learning Rate Decay (LLRD) for optimal fine-tuning of transformer and CNN architectures like `vit`, `convnextv2`.
  - Mixed Precision Training for faster execution and lower memory footprint.
  - Gradient Accumulation.
- **Data Augmentation**: MixUp and CutMix integrations for regularization.
- **Customizable Configuration**: Highly modular experiment setups using `omegaconf` (YAML config files).
- **Experiment Tracking**: Full integration with Weights & Biases logging everything from hyperparameter configs to validation metrics.

## Results

| Model | mAP | Accuracy |
| :--- | :---: | :---: |
| EfficientNetV2 Small | 0.87 | 0.815 |
| DINOv3 ViT Small Plus | 0.91 | 0.830 |
| ConvNeXtV2 Tiny | 0.94 | 0.860 |

## Project Structure

```
Plant-Disease-Classification/
├── configs/
│   └── config.yaml          # Main configuration file
├── data/
│   ├── train/               # Train data (organized by class folders)
│   └── val/                 # Val data (organized by class folders)
├── src/
│   ├── dataset.py           # Dataloaders and augmentation logic
│   ├── infer.py             # Inference script and prediction utilities
│   ├── loss.py              # Loss functions (CrossEntropy, Focal Loss)
│   ├── metrics.py           # Metric calculations
│   ├── models.py            # Model definitions and param groupings
│   ├── trainer.py           # Core training loop
│   └── utils.py             # Helpers (schedulers, seeds, config loading)
├── train.py                 # Main entrypoint for training
└── requirements.txt         # Project dependencies
```

## Quick Start

### 1. Environment Setup

It is highly recommended to use [`uv`](https://github.com/astral-sh/uv) for fast, reliable package management.

```bash
# Create a virtual environment using uv
uv venv

# Activate the environment
source .venv/bin/activate  # Linux/MacOS

# Install dependencies rapidly
uv pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your dataset is arranged in PyTorch `ImageFolder` format. Place the training data in `data/train` and validation data in `data/val`. Each subplot or leaf should be in its corresponding disease or health category folder.

```text
data/
└── train/
    ├── Apple scab/
    └── ...
```

### 3. Provide Configuration

Modify the hyperparameters, model choices, and paths inside `configs/config.yaml`.


### 4. Train the Model

Run the training pipeline:

```bash
python train.py --config configs/config.yaml
```

**Resuming Training**:
To resume from an existing checkpoint, pass the `--resume` argument:
```bash
python train.py --config configs/config.yaml --resume checkpoints/checkpoint.pth
```

To load weights for a warm start (e.g., finetuning), use:
```bash
python train.py --config configs/config.yaml --init_weights weights/pretrained.pth
```

### 5. Inference

You can run inference on a single image using the `src/infer.py` script. The script requires a serialized TorchScript model checkpoint.

```bash
# Basic inference
python src/infer.py --image_path path/to/leaf.jpg --checkpoint checkpoints/best_model.pt --image_size 384

# Inference with Test Time Augmentation (TTA)
python src/infer.py --image_path path/to/leaf.jpg --checkpoint checkpoints/best_model.pt --image_size 384 --tta
```

> **Note**: The inference script expects a `data/label_map.json` file to map class indices to disease names.

## Documentation

### Model Selection
By default, the pipeline uses `timm.create_model(...)`. You can specify any model architecture available in `timm` (e.g. `convnextv2_base`, `efficientnet_b0`, `eva02_base_patch14_448`) directly in the `config.yaml` file under `model.backbone`.

### Configuration Details
The pipeline uses `OmegaConf`. Hyperparameters such as `loss`, `optimizer`, and `augmentation` can be tweaked. For example, to enable layer-wise learning rate decay, adjust `optimizer.layer_decay` to a value `< 1.0`.

### Logging & Checkpoints
- Checkpoints are saved under the `checkpoints/` directory (customizable via `logging.checkpoint_dir`).
- Best model checkpoints (current and EMA) are tracked based on the monitored validation metric.
- When `logging.use_wandb` is true, the script initializes a Weights & Biases run, logging train/validation losses and selected metrics seamlessly.

## Model Weights
---

The trained weights are hosted on Hugging Face 
- 🔗 **[Download from Hugging Face Space Files](https://huggingface.co/spaces/)**


## Technical Report
A comprehensive report results is included in the repository.

**[View Technical Report (PDF)]()**