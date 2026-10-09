import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

"""
src/train.py
------------
Full training & evaluation engine for the Indian Spice Classification project.

Usage (from project root):
    .venv/Scripts/python src/train.py [--backbone mobilenet_v3_small] [--epochs 20] [--batch-size 32]

Outputs:
    models/best_model.pth          – weights of the epoch with the best val loss
    reports/history.json           – per-epoch metrics dictionary
    reports/training_curves.png    – 2-panel loss & metric plot
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

# ── Make src/ importable when run from project root ───────────────────────────
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import CLASS_TO_IDX, get_dataloaders
from model import build_model, freeze_backbone
from utils import AverageMeter, compute_class_weights, get_device

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = _SRC.parent
CSV_PATH     = PROJECT_ROOT / "dataset_master.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train spice classifier")
    p.add_argument("--backbone",    default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "resnet18"])
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",type=float, default=1e-2)
    p.add_argument("--num-classes", type=int, default=14)
    p.add_argument("--no-pretrain", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    return p.parse_args()


# ── One epoch ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter("loss")
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels, _ in tqdm(loader, leave=False,
                                      desc="train" if is_train else "val  "):
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Metrics
    acc  = (np.array(all_preds) == np.array(all_labels)).mean()
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss_meter.avg, acc, f1


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_history(history: dict, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Summary – Indian Spice Classifier", fontsize=14,
                 fontweight="bold")

    # Panel 1: Loss curves
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2196F3")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val Loss",   color="#F44336")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Val F1 & Accuracy
    ax2.plot(epochs, history["val_f1"],  "D-", label="Val Macro F1", color="#4CAF50")
    ax2.plot(epochs, history["val_acc"], "^-", label="Val Top-1 Acc", color="#FF9800")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation Metrics")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved training curves to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Spice Classifier – Training Run")
    print(f"{'='*60}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}  |  WD: {args.weight_decay}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_PATH, batch_size=args.batch_size
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrain,
    ).to(device)

    # Warm-up phase: backbone frozen, head only
    freeze_backbone(model, freeze=True)

    # ── Loss / Optim / Scheduler ──────────────────────────────────────────────
    class_weights = compute_class_weights(
        csv_path=CSV_PATH,
        num_classes=args.num_classes,
        class_to_idx=CLASS_TO_IDX,
        device=device,
    )
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    unfreeze_done = False

    UNFREEZE_AFTER = max(1, args.epochs // 5)   # unfreeze after ~20% of epochs

    print(f"  Backbone will unfreeze after epoch {UNFREEZE_AFTER}\n")
    header = f"{'Ep':>3} | {'TrainLoss':>9} | {'ValLoss':>8} | {'ValAcc':>7} | {'MacroF1':>8} | {'LR':>9} | {'Time':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after warm-up
        if epoch == UNFREEZE_AFTER + 1 and not unfreeze_done:
            freeze_backbone(model, freeze=False)
            # Re-create optimizer with all params
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch, eta_min=1e-6
            )
            unfreeze_done = True
            print(f"\n  [Epoch {epoch}] Backbone unfrozen — fine-tuning all layers\n")

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )
        scheduler.step()

        # Record
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(float(val_acc), 5))
        history["val_f1"].append(round(float(val_f1), 5))

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        marker  = " *" if val_loss < best_val_loss else ""

        print(
            f"{epoch:>3} | {train_loss:>9.4f} | {val_loss:>8.4f} | "
            f"{val_acc:>7.3%} | {val_f1:>8.4f} | {lr_now:>9.2e} | "
            f"{elapsed:>5.1f}s{marker}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val loss  : {best_val_loss:.4f}  (epoch {best_epoch})")
    print(f"  Best Macro F1  : {max(history['val_f1']):.4f}")
    print(f"  Best Val Acc   : {max(history['val_acc']):.3%}")
    print(f"{'='*60}\n")

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("Loading best model weights for test-set evaluation …")
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth",
                                     map_location=device, weights_only=True))
    test_loss, test_acc, test_f1 = run_epoch(
        model, test_loader, criterion, optimizer, device, is_train=False
    )
    print(f"  Test Loss   : {test_loss:.4f}")
    print(f"  Test Acc    : {test_acc:.3%}")
    print(f"  Test Macro F1: {test_f1:.4f}\n")

    history["test_loss"] = round(test_loss, 5)
    history["test_acc"]  = round(float(test_acc), 5)
    history["test_f1"]   = round(float(test_f1), 5)

    # ── Persist artefacts ─────────────────────────────────────────────────────
    hist_path = REPORTS_DIR / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  -> Saved history to {hist_path}")

    plot_history(history, REPORTS_DIR / "training_curves.png")
    print(f"  -> Saved best model to {MODELS_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
