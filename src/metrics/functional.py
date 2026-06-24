"""
Pure, deterministic segmentation metric functions.

Operates on hard predictions (after argmax / thresholding). Defines stable
empty-mask conventions so no metric is ever NaN:

    pred empty & target empty        -> dice = 1, iou = 1
    pred non-empty & target empty    -> dice = 0, iou = 0
    pred empty & target non-empty    -> dice = 0, iou = 0

Precision / recall use the analogous convention (a fully-empty pred/target pair
scores 1, any mismatch scores 0), again avoiding division-by-zero.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch


def predict_from_logits(logits: torch.Tensor, is_binary: bool,
                        threshold: float = 0.5) -> torch.Tensor:
    """Convert logits to a hard prediction map (B×H×W, long).

    binary    : sigmoid(logits) > threshold   (logits B×1×H×W)
    multiclass: argmax(logits, dim=1)          (logits B×C×H×W)
    """
    if is_binary:
        return (torch.sigmoid(logits).squeeze(1) > threshold).long()
    return logits.argmax(dim=1)


def _class_scores(pred_c: torch.Tensor, target_c: torch.Tensor) -> Dict[str, float]:
    """Scores for a single class given boolean masks (pred_c, target_c)."""
    tp = float((pred_c & target_c).sum().item())
    pred_sum = float(pred_c.sum().item())
    target_sum = float(target_c.sum().item())
    fp = pred_sum - tp
    fn = target_sum - tp

    pred_empty = pred_sum == 0.0
    target_empty = target_sum == 0.0

    if pred_empty and target_empty:
        dice, iou = 1.0, 1.0
    elif pred_empty != target_empty:
        dice, iou = 0.0, 0.0
    else:
        dice = 2.0 * tp / (2.0 * tp + fp + fn)
        iou = tp / (tp + fp + fn)

    # precision = tp / (tp + fp); recall = tp / (tp + fn) — with stable fallbacks
    precision = (tp / pred_sum) if pred_sum > 0 else (1.0 if target_empty else 0.0)
    recall = (tp / target_sum) if target_sum > 0 else (1.0 if pred_empty else 0.0)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "support": target_sum,  # foreground pixel count of this class in target
    }


def case_class_metrics(pred: torch.Tensor, target: torch.Tensor,
                       class_values: Iterable[int]) -> Dict[int, Dict[str, float]]:
    """Per-class metrics for a single case.

    Args:
        pred, target : H×W long tensors of class values.
        class_values : the class values to score (e.g. range(num_classes), or
                       {0, 1} for binary where 1 is the foreground mask value).

    Returns:
        {class_value: {dice, iou, precision, recall, support}}
    """
    out: Dict[int, Dict[str, float]] = {}
    for c in class_values:
        c = int(c)
        out[c] = _class_scores(pred == c, target == c)
    return out
