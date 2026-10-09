"""
src/model.py
------------
Model factory for the Indian Spice Classification project.

Provides:
  - build_model(backbone, num_classes, pretrained)  – returns a ready-to-train nn.Module
  - freeze_backbone(model, freeze)                  – toggles backbone weight gradients
"""

import torch
import torch.nn as nn
from torchvision import models


# ── Supported backbones ────────────────────────────────────────────────────────
_SUPPORTED = ("mobilenet_v3_small", "resnet18")


def build_model(
    backbone: str = "mobilenet_v3_small",
    num_classes: int = 14,
    pretrained: bool = True,
) -> nn.Module:
    """Construct a classification model with a custom head.

    Args:
        backbone   : Architecture name. One of 'mobilenet_v3_small' or 'resnet18'.
        num_classes: Number of output classes (default 14).
        pretrained : If True, loads ImageNet DEFAULT weights (default True).

    Returns:
        A ``torch.nn.Module`` with the final head replaced for ``num_classes``.

    Raises:
        ValueError: If an unsupported backbone is specified.
    """
    if backbone not in _SUPPORTED:
        raise ValueError(
            f"backbone must be one of {_SUPPORTED}; got '{backbone}'"
        )

    weights = "DEFAULT" if pretrained else None

    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        # Replace only the final Linear; keep BN, Hardswish, Dropout intact
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Toggle gradient computation for all layers except the final head.

    Use ``freeze=True`` for a linear-probe warm-up phase, then call again
    with ``freeze=False`` to unfreeze for full fine-tuning.

    Args:
        model : The model returned by ``build_model``.
        freeze: If True, disables gradients for backbone layers (default True).
    """
    # Identify the head parameter names so we can skip them
    if hasattr(model, "classifier"):          # MobileNetV3
        head_prefix = "classifier"
    elif hasattr(model, "fc"):                # ResNet
        head_prefix = "fc"
    else:
        head_prefix = None

    for name, param in model.named_parameters():
        if head_prefix and name.startswith(head_prefix):
            param.requires_grad = True        # head always trainable
        else:
            param.requires_grad = not freeze

    status = "frozen" if freeze else "unfrozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Backbone {status}. Trainable params: {trainable:,} / {total:,}")
