"""
Boundary overlap metrics (runnable, eval-level).

Compares the boundary band of a predicted mask against the boundary band of the
target mask (both derived via mask_to_boundary). Uses the same empty-mask
conventions as src.metrics.functional so no value is ever NaN.

This is computed from the segmentation prediction; it does NOT require a model
that predicts boundaries directly (that is the BoundaryHead skeleton).
"""

from __future__ import annotations

from typing import Dict

from src.utils.geometry import mask_to_boundary, to_numpy_bool


def _overlap_scores(pred_b, target_b) -> Dict[str, float]:
    pred_b = to_numpy_bool(pred_b)
    target_b = to_numpy_bool(target_b)
    tp = float((pred_b & target_b).sum())
    pred_sum = float(pred_b.sum())
    target_sum = float(target_b.sum())
    fp = pred_sum - tp
    fn = target_sum - tp

    pred_empty = pred_sum == 0.0
    target_empty = target_sum == 0.0
    if pred_empty and target_empty:
        return {"boundary_dice": 1.0, "boundary_iou": 1.0}
    if pred_empty != target_empty:
        return {"boundary_dice": 0.0, "boundary_iou": 0.0}
    return {
        "boundary_dice": 2.0 * tp / (2.0 * tp + fp + fn),
        "boundary_iou": tp / (tp + fp + fn),
    }


def boundary_scores(pred_mask, target_mask, width_px: int = 3) -> Dict[str, float]:
    """Boundary dice / iou between pred and target binary masks."""
    pred_boundary = mask_to_boundary(pred_mask, width_px=width_px)
    target_boundary = mask_to_boundary(target_mask, width_px=width_px)
    return _overlap_scores(pred_boundary, target_boundary)
