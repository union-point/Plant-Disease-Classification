import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score

from dataset import get_transforms


def get_tta_transforms(image_size):
    return [
        T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.RandomHorizontalFlip(p=1.0),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        T.Compose(
            [
                T.Resize(int(image_size * 1.1), antialias=True),
                T.CenterCrop(image_size),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    ]


def evaluate(model, val_loader, device=None, use_tta=False, image_size=384):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    if use_tta:
        tta_transforms = get_tta_transforms(image_size)

    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_tta:
                tta_batches = []

                for transform in tta_transforms:
                    augmented = torch.stack([transform(img.cpu()) for img in images])
                    tta_batches.append(augmented)

                tta_batches = torch.stack(tta_batches).to(device)

                outputs = []
                for tta_batch in tta_batches:
                    out = model(tta_batch)  # [batch, num_classes]
                    outputs.append(out)

                outputs = torch.stack(outputs).mean(dim=0)

            else:
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds = np.argmax(all_probs, axis=1)
    acc = accuracy_score(all_labels, preds)

    num_classes = all_probs.shape[1]
    y_true_bin = np.zeros((len(all_labels), num_classes))
    y_true_bin[np.arange(len(all_labels)), all_labels] = 1

    per_class_ap = []
    for i in range(num_classes):
        if y_true_bin[:, i].sum() > 0:
            ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
            per_class_ap.append(ap)

    mAP = np.mean(per_class_ap)

    return acc, mAP, all_probs, all_labels


def predict_disease(
    model, image, idx_to_disease, image_size=384, use_tta=False, device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    if use_tta:
        transforms = get_tta_transforms(image_size)
        tensors = [transform(image).unsqueeze(0) for transform in transforms]
        batch = torch.cat(tensors, dim=0).to(device)

        with torch.inference_mode():
            outputs = model(batch)
            output = outputs.mean(dim=0, keepdim=True)

    else:
        transform = get_transforms(image_size, is_train=False)
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(tensor)

    probs = output.softmax(dim=1)
    disease_name = idx_to_disease[probs.argmax(dim=1).item()]

    return disease_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a plant disease image"
    )
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--image_size", type=str, default=384, help="Size of input image"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint "
    )
    parser.add_argument("--tta", action="store_true", help="Use test time augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(args.checkpoint).to(device)
    model.eval()
    print(args.tta)
    # load label map
    data_dir = Path("data")
    label_map_path = data_dir / "label_map.json"
    with open(label_map_path) as f:
        label_map = json.load(f)
    idx_to_disease = {int(v): k for k, v in label_map.items()}

    image = Image.open(args.image_path).convert("RGB")

    result = predict_disease(
        model,
        image,
        image_size=args.image_size,
        idx_to_disease=idx_to_disease,
        use_tta=args.tta,
    )
    print(f"Disease: {result}")
