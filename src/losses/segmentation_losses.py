"""
Segmentation loss modules.

The Dice term comes from segmentation-models-pytorch. This module is the ONLY
place that imports smp losses, so ``SegModule`` stays free of any direct smp
dependency (loss is obtained through ``src.losses.factory.build_loss``).

All losses are computed from raw logits — models must NOT apply sigmoid/softmax
in their forward pass.
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceWithAuxLoss(nn.Module):
    """Dice + auxiliary cross-entropy loss, computed from logits.

    forward(logits, targets) returns the 3-tuple ``(total, dice, aux)`` where
    ``total = dice_weight * dice + aux_weight * aux``. The raw (unweighted)
    ``dice`` and ``aux`` terms are returned so callers can log them separately.

    Args:
        mode        : "binary" (logits B×1×H×W, BCEWithLogitsLoss aux) or
                      "multiclass" (logits B×C×H×W, CrossEntropyLoss aux).
        dice_weight : weight on the Dice term.
        aux_weight  : weight on the auxiliary CE/BCE term.
        from_logits : passed to smp DiceLoss (True → it applies the activation
                      internally; keep True since models output logits).
    """

    def __init__(self, mode: str, dice_weight: float = 1.0,
                 aux_weight: float = 1.0, from_logits: bool = True):
        super().__init__()
        if mode not in ("binary", "multiclass"):
            raise ValueError(f"mode must be 'binary' or 'multiclass', got {mode!r}")
        self.mode = mode
        self.dice_weight = float(dice_weight)
        self.aux_weight = float(aux_weight)
        self.dice = smp.losses.DiceLoss(mode=mode, from_logits=from_logits)
        self.aux = nn.BCEWithLogitsLoss() if mode == "binary" else nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        dice = self.dice(logits, targets)
        if self.mode == "binary":
            # BCEWithLogitsLoss expects float targets shaped like the logits.
            aux = self.aux(logits.squeeze(1).float(), targets.float())
        else:
            aux = self.aux(logits, targets)
        total = self.dice_weight * dice + self.aux_weight * aux
        return total, dice, aux
