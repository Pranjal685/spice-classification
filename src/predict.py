"""
src/predict.py
--------------
Two-Tier Cascaded Inference Engine for the Indian Spice Classification project.

Tier 1: Local MobileNetV3 CNN with 75% confidence gating.
Tier 2: Gemini Vision (gemini-2.5-flash) zero-shot fallback for uncertain or OOD images.

Usage:
    python src/predict.py --image path/to/spice.jpg [--threshold 0.75] [--force-fallback]
"""

import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# ── resolve local imports --------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import build_model
from utils import get_device
from fallback import SpiceFallbackEngine

# ── paths ------------------------------------------------------------------
PROJECT_ROOT    = _SRC.parent
MODEL_PATH      = PROJECT_ROOT / "models" / "best_model.pth"
CLASS_IDX_PATH  = PROJECT_ROOT / "data" / "class_indices.json"
KNOWLEDGE_PATH  = PROJECT_ROOT / "data" / "spice_knowledge.json"

# ── ImageNet normalisation -------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ── loaders ----------------------------------------------------------------
def _load_class_maps():
    """Load class_to_idx and idx_to_class from persisted JSON."""
    with open(CLASS_IDX_PATH) as f:
        data = json.load(f)
    class_to_idx = data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    return class_to_idx, idx_to_class


def _load_knowledge_base():
    """Load the spice knowledge dictionary."""
    with open(KNOWLEDGE_PATH) as f:
        return json.load(f)


def _load_model(device, num_classes):
    """Load the best checkpoint in eval mode."""
    model = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model


# ── Tier-1: local CNN inference --------------------------------------------
def predict_image(image_path, model, device, idx_to_class, threshold=0.75):
    """Run single-image Tier-1 inference and return structured results.

    Returns:
        dict with keys:
            tier          : 'TIER_1_LOCAL_CNN'
            status        : 'CONFIDENT_PREDICTION' or 'LOW_CONFIDENCE'
            top1_class    : str class label
            top1_prob     : float probability
            top3          : list of (class_label, probability) tuples
            above_threshold: bool
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = _INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Top-3
    top3_idx  = probs.argsort()[::-1][:3]
    top3      = [(idx_to_class[int(i)], float(probs[i])) for i in top3_idx]

    top1_class = top3[0][0]
    top1_prob  = top3[0][1]
    above      = top1_prob >= threshold

    return {
        "tier":           "TIER_1_LOCAL_CNN",
        "status":         "CONFIDENT_PREDICTION" if above else "LOW_CONFIDENCE",
        "top1_class":     top1_class,
        "top1_prob":      top1_prob,
        "top3":           top3,
        "above_threshold": above,
    }


# ── terminal output formatters --------------------------------------------
_SEP   = "=" * 64
_THIN  = "-" * 64

def _print_tier1_confident(result, knowledge_base):
    """Print the rich Tier-1 spice knowledge card for a confident prediction."""
    cls  = result["top1_class"]
    prob = result["top1_prob"]
    info = knowledge_base.get(cls, {})

    print()
    print(_SEP)
    print("  [TIER 1 - CONFIDENT PREDICTION] Spice Identified!")
    print(_SEP)
    print(f"  Tier            : TIER_1_LOCAL_CNN")
    print(f"  Status          : CONFIDENT_PREDICTION")
    print(f"  Confidence      : {prob:.1%}")
    print(_THIN)
    print(f"  Identified Spice     : {info.get('display_name', cls)}")
    print(f"  Scientific Name      : {info.get('scientific_name', 'N/A')}")
    print(f"  Flavor Profile       : {info.get('flavor_profile', 'N/A')}")
    print(f"  Common Culinary Uses : {info.get('common_uses', 'N/A')}")
    print(f"  Visual Features      : {info.get('key_visual_features', 'N/A')}")
    print(_THIN)

    # Top-3 probabilities
    print("  Top-3 Predictions:")
    for rank, (c, p) in enumerate(result["top3"], 1):
        display = knowledge_base.get(c, {}).get("display_name", c)
        print(f"    {rank}. {display:<35s} {p:>6.1%}")
    print(_SEP)


def _print_tier2_result(result, t1_result, knowledge_base):
    """Print the Tier-2 Gemini fallback result card."""
    print()
    print(_SEP)
    print("  [TIER 2 - GEMINI VISION FALLBACK]")
    print(_SEP)
    print(f"  Tier            : TIER_2_GEMINI_FALLBACK")

    # Show Tier-1 context if available
    if t1_result:
        t1_top  = t1_result["top1_class"]
        t1_conf = t1_result["top1_prob"]
        t1_disp = knowledge_base.get(t1_top, {}).get("display_name", t1_top)
        print(f"  Tier-1 Best Guess: {t1_disp} at {t1_conf:.1%} (below threshold)")
        print()

    # Check for errors
    if result.get("error"):
        print(f"  [ERROR] {result.get('error_reason', 'Unknown')}")
        print(f"  Assessment: {result.get('confidence_assessment', 'N/A')}")
        print(f"  Action:     {result.get('action_recommendation', 'N/A')}")
        print(_SEP)
        return

    is_spice = result.get("is_spice", False)

    if not is_spice:
        print(f"  Status          : NOT_A_SPICE")
        print(f"  Assessment      : {result.get('confidence_assessment', 'N/A')}")
        print(_THIN)
        print(f"  Action: {result.get('action_recommendation', 'N/A')}")
        print(_SEP)
        return

    # It's a spice
    in_scope = result.get("is_in_dataset_scope", False)
    status   = "IN_SCOPE_SPICE" if in_scope else "OUT_OF_DISTRIBUTION_SPICE"
    print(f"  Status          : {status}")
    print(_THIN)
    print(f"  Identified Spice     : {result.get('identified_name', 'Unknown')}")
    print(f"  Scientific Name      : {result.get('scientific_name', 'N/A')}")
    print(f"  In Dataset Scope     : {'Yes' if in_scope else 'No (OOD)'}")
    print(f"  Assessment           : {result.get('confidence_assessment', 'N/A')}")
    print(f"  Flavor Profile       : {result.get('flavor_profile', 'N/A')}")
    print(f"  Common Culinary Uses : {result.get('common_uses', 'N/A')}")
    print(f"  Recommendation       : {result.get('action_recommendation', 'N/A')}")
    print(_SEP)


def _print_tier1_low_with_fallback(t1_result, t2_result, knowledge_base):
    """Print combined card when Tier-1 is low-confidence and Tier-2 is invoked."""
    cls  = t1_result["top1_class"]
    prob = t1_result["top1_prob"]
    display = knowledge_base.get(cls, {}).get("display_name", cls)

    print()
    print(_SEP)
    print("  [LOW CONFIDENCE] Tier-1 uncertain -> invoking Tier-2 Gemini")
    print(_SEP)
    print(f"  Tier-1 Top Prediction : {display} at {prob:.1%}")
    print(f"  Threshold             : Not met (< 75%)")
    print()
    print("  Tier-1 Top-3 Candidates:")
    for rank, (c, p) in enumerate(t1_result["top3"], 1):
        d = knowledge_base.get(c, {}).get("display_name", c)
        print(f"    {rank}. {d:<35s} {p:>6.1%}")
    print()
    print("  Escalating to Tier-2 Gemini Vision API ...")
    print(_THIN)

    _print_tier2_result(t2_result, t1_result, knowledge_base)


# ── CLI entry point --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Two-Tier Cascaded Spice Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python src/predict.py --image "data/raw/Cumin 58/photo.jpg"\n'
            '  python src/predict.py --image img.jpg --threshold 0.85\n'
            '  python src/predict.py --image mystery.jpg --force-fallback\n'
        ),
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the spice image to classify")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold for Tier-1 gate (default: 0.75)")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Skip Tier-1 and go directly to Tier-2 Gemini fallback")
    args = parser.parse_args()

    device = get_device()
    class_to_idx, idx_to_class = _load_class_maps()
    knowledge_base = _load_knowledge_base()
    num_classes = len(class_to_idx)

    # ── Force fallback mode ───────────────────────────────────────────────
    if args.force_fallback:
        print(f"\n[--force-fallback] Skipping Tier-1, going directly to Gemini Vision.")
        print(f"Image: {args.image}")
        engine = SpiceFallbackEngine()
        print(f"Engine: {engine}")
        t2_result = engine.identify_unknown_image(args.image)
        _print_tier2_result(t2_result, None, knowledge_base)
        return t2_result

    # ── Tier-1: Local CNN ─────────────────────────────────────────────────
    print(f"\nLoading model from {MODEL_PATH} ...")
    model = _load_model(device, num_classes)
    print(f"Device: {device}  |  Classes: {num_classes}  |  Threshold: {args.threshold:.0%}")
    print(f"Image:  {args.image}")

    t1_result = predict_image(args.image, model, device, idx_to_class, args.threshold)
    t1_result["tier"] = "TIER_1_LOCAL_CNN"

    # ── Decision Gate ─────────────────────────────────────────────────────
    if t1_result["above_threshold"]:
        # Tier-1 confident -> print knowledge card
        _print_tier1_confident(t1_result, knowledge_base)
        return t1_result

    # ── Tier-2: Gemini Fallback ───────────────────────────────────────────
    engine = SpiceFallbackEngine()
    t2_result = engine.identify_unknown_image(args.image)
    t2_result["tier"] = "TIER_2_GEMINI_FALLBACK"

    _print_tier1_low_with_fallback(t1_result, t2_result, knowledge_base)

    return t2_result


if __name__ == "__main__":
    main()
