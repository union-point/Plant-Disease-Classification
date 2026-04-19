import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, label_map=None, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of the split.
            label_map (dict, optional): A dictionary mapping disease names to integers.
                             Crucial for consistency across splits.
            transform (callable, optional): PyTorch transforms.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.plant_labels = []

        self.plants = [
            "apple",
            "banana",
            "bean",
            "bell pepper",
            "blueberry",
            "basil",
            "broccoli",
            "cabbage",
            "cauliflower",
            "celery",
            "cherry",
            "citrus",
            "coffee",
            "corn",
            "cucumber",
            "garlic",
            "ginger",
            "grape",
            "lettuce",
            "maple",
            "peach",
            "plum",
            "potato",
            "raspberry",
            "rice",
            "soybean",
            "squash",
            "strawberry",
            "tobacco",
            "tomato",
            "wheat",
            "zucchini",
        ]
        self.plants.sort(key=len, reverse=True)

        if not self.root_dir.exists():
            return

        if label_map is None:
            self.disease_to_idx = self._build_label_map()
        else:
            self.disease_to_idx = label_map

        for folder_name in sorted([d for d in self.root_dir.iterdir() if d.is_dir()]):
            disease, plant = self._split_plant_disease(folder_name)

            if disease not in self.disease_to_idx:
                print(
                    f"WARNING: Skipping '{folder_name.name}': Disease '{disease}' not found in label_map"
                )
                continue

            disease_idx = self.disease_to_idx[disease]

            for img_path in folder_name.glob("**/*"):
                if img_path.is_file() and img_path.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ]:
                    self.image_paths.append(str(img_path))
                    self.labels.append(disease_idx)
                    self.plant_labels.append(plant)

        self.classes = list(self.disease_to_idx.keys())

    def _build_label_map(self):
        all_diseases = set()

        for folder in sorted([d for d in self.root_dir.iterdir() if d.is_dir()]):
            folder_name = folder.name.lower()
            for plant in self.plants:
                if folder_name.startswith(plant):
                    disease_name = folder_name[len(plant) :].strip()
                    all_diseases.add(disease_name)
                    break

        return {disease: i for i, disease in enumerate(sorted(list(all_diseases)))}

    def _split_plant_disease(self, folder):
        for plant in self.plants:
            folder_name = folder.name.lower()
            if folder_name.startswith(plant):
                disease = folder_name[len(plant) :].strip()
                return disease, plant

        return None, None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=384, is_train=True):
    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def build_weighted_sampler(dataset, max_weight=10.0):
    if not hasattr(dataset, "labels") or not hasattr(dataset, "plant_labels"):
        raise ValueError("Dataset must have 'labels' and 'plant_labels'")

    if len(dataset) == 0:
        return None

    disease = torch.tensor(dataset.labels, dtype=torch.long)
    _, plant_indices = np.unique(dataset.plant_labels, return_inverse=True)
    plant = torch.tensor(plant_indices, dtype=torch.long)

    disease_counts = torch.bincount(disease)

    pairs = torch.stack([disease, plant], dim=1)
    _, group_id = torch.unique(pairs, return_inverse=True, dim=0)
    group_counts = torch.bincount(group_id)

    d_count = disease_counts[disease]
    g_count = group_counts[group_id]

    weights = 1.0 / torch.sqrt(d_count.float() * g_count.float())

    if max_weight:
        weights = torch.clamp(weights, max=max_weight)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


def get_dataloaders(config):
    train_dir = Path(config.data.train_dir)
    val_dir = Path(config.data.val_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # load existing if dont exist dataset wil build from training automatically
    label_map_path = train_dir.parent / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)
    else:
        label_map = None

    # create datasets with consistent label_map
    train_dataset = PlantDiseaseDataset(
        config.data.train_dir,
        label_map=label_map,
        transform=get_transforms(config.data.image_size, is_train=True),
    )
    val_dataset = PlantDiseaseDataset(
        config.data.val_dir,
        label_map=train_dataset.disease_to_idx,
        transform=get_transforms(config.data.image_size, is_train=False),
    )

    # save label_map for future
    with open(label_map_path, "w") as f:
        json.dump(train_dataset.disease_to_idx, f, indent=2)

    if len(train_dataset) == 0:
        print("Warning: No train data found. Dataloader might fail.")

    num_classes = len(train_dataset.classes) if len(train_dataset) > 0 else 0

    train_sampler = None
    if config.data.weighted_sampling:
        train_sampler = build_weighted_sampler(
            train_dataset, max_weight=config.data.max_weight
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True if len(train_dataset) > config.data.batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    return train_loader, val_loader, num_classes
