"""
Deterministic clinical morphology metrics, computed from a predicted mask.

First version (Phase 9b): evaluation-only, derived from the predicted mask via
src.metrics.morphology. Values are in PIXEL units (unit = "pixel"); no physical
unit conversion is performed. These are non-differentiable and must never enter
a training loss.
"""

from __future__ import annotations

from typing import Dict, Sequence

from src.metrics.morphology import area, area_ratio, perimeter

DEFAULT_METRICS = ("area", "perimeter", "area_ratio")
PIXEL_UNIT = "pixel"


def compute_class_clinical(class_mask, metrics: Sequence[str] = DEFAULT_METRICS,
                           total_pixels: float | None = None) -> Dict[str, float]:
    """Compute the requested morphology metrics for a single binary class mask.

    Args:
        class_mask   : H×W boolean / {0,1} mask of one class' prediction.
        metrics      : subset of {"area", "perimeter", "area_ratio"}.
        total_pixels : denominator for area_ratio (defaults to mask size).
    """
    out: Dict[str, float] = {}
    for name in metrics:
        if name == "area":
            out["area"] = area(class_mask)
        elif name == "perimeter":
            out["perimeter"] = perimeter(class_mask)
        elif name == "area_ratio":
            out["area_ratio"] = area_ratio(class_mask, total_pixels)
        else:
            raise ValueError(
                f"Unknown clinical metric '{name}'. "
                f"Supported: {sorted(DEFAULT_METRICS)}"
            )
    return out
