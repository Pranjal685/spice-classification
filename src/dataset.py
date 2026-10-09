"""
src/dataset.py
--------------
PyTorch ingestion pipeline for the Indian Spice Classification project.

Provides:
  - CLASS_TO_IDX / IDX_TO_CLASS  – deterministic bidirectional label maps
  - SpiceDataset                 – torch.utils.data.Dataset subclass
  - get_transforms(split)        – augmentation pipelines per split
  - get_dataloaders(batch_size)  – factory returning train/val/test loaders
"""

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CSV_PATH       = PROJECT_ROOT / "dataset_master.csv"
CLASS_IDX_PATH = PROJECT_ROOT / "data" / "class_indices.json"

# ── Bidirectional label maps ───────────────────────────────────────────────────
def _build_label_maps(csv_path: Path = CSV_PATH):
    """Derive sorted, deterministic label maps from dataset_master.csv."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["class_label"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c

CLASS_TO_IDX, IDX_TO_CLASS = _build_label_maps()

# Persist to disk for inference scripts
CLASS_IDX_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CLASS_IDX_PATH, "w") as f:
    json.dump({"class_to_idx": CLASS_TO_IDX,
               "idx_to_class": {str(k): v for k, v in IDX_TO_CLASS.items()}}, f, indent=2)


# ── Augmentation Pipelines ─────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str) -> transforms.Compose:
    """Return the appropriate torchvision transform pipeline for a given split.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        A ``torchvision.transforms.Compose`` object.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])


# ── Custom Dataset ─────────────────────────────────────────────────────────────
class SpiceDataset(Dataset):
    """PyTorch Dataset for Indian Spice image classification.

    Args:
        csv_path  : Path to dataset_master.csv.
        split     : One of 'train', 'val', or 'test'.
        transform : A callable transform applied to each PIL image.
                    If None, ``get_transforms(split)`` is used automatically.
    """

    def __init__(self, csv_path: Path = CSV_PATH, split: str = "train",
                 transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        df = pd.read_csv(csv_path)

        # Filter by split (and validity flag if column exists)
        mask = df["split"] == split
        if "is_valid" in df.columns:
            mask &= df["is_valid"].astype(bool)

        self.df        = df[mask].reset_index(drop=True)
        self.split     = split
        self.transform = transform if transform is not None else get_transforms(split)
        self.root      = PROJECT_ROOT

    # ── dunder helpers ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """Return (image_tensor, label_idx, image_id).

        Args:
            idx: Integer row index.

        Returns:
            Tuple of (torch.FloatTensor [3,224,224], int label index, str image_id).
        """
        row       = self.df.iloc[idx]
        img_path  = self.root / row["file_path"]
        image_id  = img_path.stem                  # filename without extension
        label_idx = CLASS_TO_IDX[row["class_label"]]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label_idx, image_id


# ── DataLoaders Factory ────────────────────────────────────────────────────────
def get_dataloaders(csv_path: Path = CSV_PATH,
                    batch_size: int = 32,
                    num_workers: int = 0):
    """Instantiate train, val, and test DataLoaders.

    Args:
        csv_path    : Path to dataset_master.csv.
        batch_size  : Number of samples per batch (default 32).
        num_workers : Subprocesses for data loading (default 0 = main process).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = SpiceDataset(csv_path=csv_path, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),   # avoid partial batches during training
        )

    return loaders["train"], loaders["val"], loaders["test"]
