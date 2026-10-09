"""
src/visualize_batch.py
-----------------------
Loads one batch from train_loader, denormalizes images, and saves a
4×4 grid with class-name titles to reports/sample_batch.png.

Usage:
    python src/visualize_batch.py
"""

from pathlib import Path
import sys

# Ensure src/ is on the path so `from dataset import ...` works when the
# script is launched from the project root as: python src/visualize_batch.py
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import get_dataloaders, IDX_TO_CLASS

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = REPORTS_DIR / "sample_batch.png"

ROWS, COLS   = 4, 4
N_IMAGES     = ROWS * COLS          # 16

# ImageNet stats used during normalization (to invert)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization and clamp to [0, 1]."""
    return (tensor * _STD + _MEAN).clamp(0, 1)


def main():
    print("Loading train DataLoader …")
    train_loader, _, _ = get_dataloaders(batch_size=N_IMAGES)

    images, labels, image_ids = next(iter(train_loader))
    # images : (N, 3, 224, 224) normalised tensors
    # labels : (N,) integer indices
    # image_ids: tuple of N strings

    print(f"Batch shape  : {images.shape}")
    print(f"Label indices: {labels.tolist()}")

    fig, axes = plt.subplots(ROWS, COLS, figsize=(16, 16))
    fig.suptitle("Training Batch — Augmented Spice Samples", fontsize=18,
                 fontweight="bold", y=0.995)

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis("off")
            continue

        img  = denormalize(images[i]).permute(1, 2, 0).numpy()   # HWC
        cls  = IDX_TO_CLASS[labels[i].item()]
        name = cls.replace("_", " ").title()

        ax.imshow(img)
        ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 4×4 grid to: {OUT_PATH}")


if __name__ == "__main__":
    main()
