# Plant Disease Classification

An AI system for plant disease classification from a leaf/plant image: accepts images of any host species and predicts the disease regardless of host.


- **Live API:** [unionpoint/plant-disease-classification](https://huggingface.co/spaces/unionpoint/plant-disease-classification)

## Results

| Model | mAP | Accuracy | Params (M) |  links |
| ---- | ---- | ---- | ---- | ---- |
| EfficientNetV2 Small | 0.87 | 0.81 | 21.5 |[ckpt](https://huggingface.co/unionpoint/tf_efficientnetv2_s.ft_plantdoc_384) |
| DINOv3 ViT Small Plus | 0.91 | 0.83 | 28.7 |[ckpt](https://huggingface.co/unionpoint/vit_small_plus_patch16_dinov3.ft_plantdoc_384) |
| ConvNeXtV2 Tiny | **0.94** | **0.86** | 28.6 |[ckpt](https://huggingface.co/unionpoint/convnextv2_tiny.ft_plantdoc_384) |
## Project Structure

```
Plant-Disease-Classification/
├── configs/       # train configuration files
├── data/
│   ├── train/               # Train data (organized by class folders)
│   └── val/                 # Val data (organized by class folders)
├── src/
│   ├── dataset.py           # Dataloaders and augmentation logic
│   ├── infer.py             # Inference script 
│   ├── loss.py              # Loss functions (CrossEntropy, Focal Loss)
│   ├── metrics.py           # Metric calculations
│   ├── models.py            # Model definitions and param groupings
│   ├── trainer.py           # Core training loop
│   └── utils.py             # Helpers
├── train.py                 # Main entrypoint for training
└── requirements.txt         # Project dependencies
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/union-point/Plant-Disease-Classification.git

uv pip install -r requirements.txt
```

### 2. Provide Configuration

Open `configs/config.yaml`, update the paths and specify the required hyperparameters, or load one of exiting configs.

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

### API

Run locally:

```bash
uv run -m api.main
```
## Documentation

### Model Selection
By default, the pipeline uses `timm.create_model(...)`. You can specify any model architecture available in `timm` (e.g. `convnextv2_base`, `efficientnet_b0`, `eva02_base_patch14_448`) directly in the `config.yaml` file under `model.backbone`.

### Logging & Checkpoints
- Checkpoints are saved under the `checkpoints/` directory (customizable via `logging.checkpoint_dir`).
- Best model checkpoints (current and EMA) are tracked based on the monitored validation metric.
- When `logging.use_wandb` is true, the script initializes a Weights & Biases run, logging train/validation losses and selected metrics seamlessly.


