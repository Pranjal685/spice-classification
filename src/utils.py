"""
src/utils.py
------------
Shared utilities for the Indian Spice Classification project.

Provides:
  - get_device()               – returns the best available torch.device
  - compute_class_weights(...) – inverse-frequency balanced loss weights
  - AverageMeter               – simple running-mean tracker for scalars
"""

from pathlib import Path

import pandas as pd
import torch


# ── Device ─────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Class-balanced loss weights ────────────────────────────────────────────────
def compute_class_weights(
    csv_path: str | Path,
    num_classes: int,
    class_to_idx: dict,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``nn.CrossEntropyLoss``.

    Formula: w_c = (N_total) / (C * N_c), then normalised so mean(w) == 1.

    Args:
        csv_path    : Path to dataset_master.csv.
        num_classes : Total number of label classes.
        class_to_idx: Dict mapping class-label strings to integer indices.
        device      : Target torch device.

    Returns:
        A ``torch.FloatTensor`` of shape ``(num_classes,)`` on ``device``.
    """
    df    = pd.read_csv(csv_path)
    train = df[df["split"] == "train"]

    counts  = train["class_label"].value_counts()        # Series
    weights = torch.zeros(num_classes, dtype=torch.float32)

    n_total = len(train)
    for label, idx in class_to_idx.items():
        n_c = counts.get(label, 1)                       # avoid div-by-zero
        weights[idx] = n_total / (num_classes * n_c)

    # Normalise so mean weight == 1  (keeps LR intuition stable)
    weights = weights / weights.mean()

    return weights.to(device)


# ── Running-mean helper ────────────────────────────────────────────────────────
class AverageMeter:
    """Tracks a running arithmetic mean of a scalar (e.g. batch loss)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0
        self.avg   = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
