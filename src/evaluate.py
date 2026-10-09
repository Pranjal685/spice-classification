"""
src/evaluate.py
---------------
Comprehensive model evaluation & diagnostic suite for the Indian Spice
Classification project.

Computes:
  - Standard test accuracy and macro F1-score
  - Coverage rate, gated accuracy, and rejection rate at a configurable
    confidence threshold (default 0.75)
  - Normalised confusion matrix saved to reports/confusion_matrix.png
  - Fine-grained sub-analysis of the four chilli cultivars

Usage:
    python src/evaluate.py [--threshold 0.75]
"""

import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── resolve local imports --------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset import CLASS_TO_IDX, IDX_TO_CLASS, SpiceDataset, get_transforms
from model import build_model
from utils import get_device

# ── paths ------------------------------------------------------------------
PROJECT_ROOT = _SRC.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "best_model.pth"
REPORTS_DIR  = PROJECT_ROOT / "reports"
CSV_PATH     = PROJECT_ROOT / "dataset_master.csv"
REPORTS_DIR.mkdir(exist_ok=True)

NUM_CLASSES  = len(CLASS_TO_IDX)
CHILLI_CLASSES = ["chilli_byadgi", "chilli_guntur", "chilli_lavangi", "chilli_sankeshwari"]


# ── helpers ----------------------------------------------------------------
def _load_model(device: torch.device) -> torch.nn.Module:
    """Load the best checkpoint and return the model in eval mode."""
    model = build_model("mobilenet_v3_small", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


def _collect_predictions(model, dataset, device, batch_size=16):
    """Run inference over the entire dataset and return parallel lists.

    Returns:
        all_labels  : list[int]   – ground-truth indices
        all_preds   : list[int]   – predicted indices
        all_probs   : np.ndarray  – (N, C) probability matrix
        all_ids     : list[str]   – image identifiers
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    all_labels, all_preds, all_probs, all_ids = [], [], [], []

    with torch.no_grad():
        for images, labels, img_ids in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            preds = probs.argmax(axis=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_ids.extend(img_ids)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_labels, all_preds, all_probs, all_ids


# ── confusion matrix plotting ----------------------------------------------
def _plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save a normalised confusion matrix heatmap."""
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # handle classes with 0 test samples

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Normalised Confusion Matrix - Indian Spice Classifier", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> Confusion matrix saved to {save_path}")


# ── chilli sub-analysis ----------------------------------------------------
def _chilli_sub_analysis(y_true, y_pred, all_probs, class_names):
    """Print diagnostic breakdown confined to the four chilli cultivars."""
    chilli_idxs = [CLASS_TO_IDX[c] for c in CHILLI_CLASSES if c in CLASS_TO_IDX]

    # Filter to samples whose TRUE label is a chilli cultivar
    mask = np.isin(y_true, chilli_idxs)
    if mask.sum() == 0:
        print("\n  [Chilli Analysis] No chilli samples found in the test set.")
        return

    y_true_ch = np.array(y_true)[mask]
    y_pred_ch = np.array(y_pred)[mask]

    chilli_names = [class_names[i] for i in chilli_idxs]
    cm = confusion_matrix(y_true_ch, y_pred_ch, labels=chilli_idxs)

    sep = "-" * 68
    print(f"\n{sep}")
    print("  CHILLI CULTIVAR FINE-GRAINED SUB-ANALYSIS")
    print(sep)
    print(f"  Total chilli test samples: {mask.sum()}")
    print(f"  Correct predictions:       {(y_true_ch == y_pred_ch).sum()}")
    print(f"  Intra-chilli accuracy:     {accuracy_score(y_true_ch, y_pred_ch):.1%}")
    print()

    # Print mini confusion matrix
    header = "  True \\ Pred".ljust(24) + "".join(c.replace("chilli_","").ljust(14) for c in CHILLI_CLASSES)
    print(header)
    print("  " + "-" * len(header))
    for i, row_idx in enumerate(chilli_idxs):
        row_name = CHILLI_CLASSES[i].replace("chilli_", "").ljust(20)
        row_vals = "".join(str(cm[i, j]).center(14) for j in range(len(chilli_idxs)))
        print(f"  {row_name}  {row_vals}")

    # Check for cross-chilli vs cross-category errors
    cross_chilli = 0
    cross_category = 0
    for t, p in zip(y_true_ch, y_pred_ch):
        if t != p:
            if p in chilli_idxs:
                cross_chilli += 1
            else:
                cross_category += 1

    total_errors = cross_chilli + cross_category
    print(f"\n  Misclassification breakdown:")
    print(f"    Errors within chilli group:   {cross_chilli}")
    print(f"    Errors to other categories:   {cross_category}")
    if total_errors > 0:
        pct = cross_chilli / total_errors * 100
        print(f"    -> {pct:.0f}% of chilli errors are intra-cultivar confusion")
    print(sep)


# ── main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate best spice classifier model")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold for gated metrics (default: 0.75)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    threshold = args.threshold

    # Ordered class names (index-aligned)
    class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]

    print("=" * 68)
    print("  Indian Spice Classifier - Comprehensive Evaluation")
    print("=" * 68)
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Device      : {device}")
    print(f"  Threshold   : {threshold:.0%}")
    print(f"  Num Classes : {NUM_CLASSES}")
    print("=" * 68)

    # Load
    model   = _load_model(device)
    dataset = SpiceDataset(csv_path=CSV_PATH, split="test")
    print(f"\n  Test samples: {len(dataset)}")

    # Predict
    y_true, y_pred, all_probs, all_ids = _collect_predictions(
        model, dataset, device, batch_size=args.batch_size
    )
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── 1. Standard Metrics ───────────────────────────────────────────────
    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 68}")
    print("  1. OVERALL TEST METRICS")
    print(f"{'=' * 68}")
    print(f"  Test Accuracy    : {test_acc:.3%}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    # ── 2. Confidence-Gated Metrics ───────────────────────────────────────
    top1_probs = all_probs[np.arange(len(y_pred)), y_pred]  # P(predicted class)
    confident_mask = top1_probs >= threshold

    n_total    = len(y_pred)
    n_covered  = confident_mask.sum()
    n_rejected = n_total - n_covered

    coverage_rate = n_covered / n_total
    rejection_rate = n_rejected / n_total
    gated_acc = accuracy_score(y_true[confident_mask], y_pred[confident_mask]) if n_covered > 0 else 0.0

    print(f"\n{'=' * 68}")
    print(f"  2. CONFIDENCE-GATED METRICS (threshold >= {threshold:.0%})")
    print(f"{'=' * 68}")
    print(f"  Coverage Rate    : {coverage_rate:.1%}  ({n_covered}/{n_total} images above threshold)")
    print(f"  Gated Accuracy   : {gated_acc:.3%}  (accuracy on covered subset only)")
    print(f"  Rejection Rate   : {rejection_rate:.1%}  ({n_rejected}/{n_total} images below threshold)")

    # List rejected samples
    if n_rejected > 0:
        print(f"\n  Rejected samples (confidence < {threshold:.0%}):")
        rejected_indices = np.where(~confident_mask)[0]
        for idx in rejected_indices:
            pred_cls  = class_names[y_pred[idx]]
            true_cls  = class_names[y_true[idx]]
            conf      = top1_probs[idx]
            status    = "CORRECT" if y_pred[idx] == y_true[idx] else "WRONG"
            print(f"    [{status}] {all_ids[idx]}: pred={pred_cls} ({conf:.1%}), true={true_cls}")

    # ── 3. Per-class classification report ────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  3. PER-CLASS CLASSIFICATION REPORT")
    print(f"{'=' * 68}")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=3,
    )
    print(report)

    # ── 4. Confusion Matrix ──────────────────────────────────────────────
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    _plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # ── 5. Chilli sub-analysis ───────────────────────────────────────────
    _chilli_sub_analysis(y_true.tolist(), y_pred.tolist(), all_probs, class_names)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 68}")
    print(f"  Test Accuracy      : {test_acc:.3%}")
    print(f"  Macro F1           : {macro_f1:.4f}")
    print(f"  Coverage (>={threshold:.0%})  : {coverage_rate:.1%}")
    print(f"  Gated Accuracy     : {gated_acc:.3%}")
    print(f"  Rejection Rate     : {rejection_rate:.1%}")
    print(f"  Confusion Matrix   : {cm_path}")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
