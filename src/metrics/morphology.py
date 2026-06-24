"""
Deterministic morphology metrics computed from a binary prediction mask.

All values are in PIXEL units. No physical-unit (mm/mm^2) conversion is done
here — if pixel spacing is ever supplied it must be handled by the caller and
clearly labelled; this module never fabricates a unit conversion.

These are non-differentiable NumPy metrics intended for EVALUATION only; they
must never be used inside a training loss.
"""

from __future__ import annotations

import numpy as np

from src.utils.geometry import to_numpy_bool


def area(mask) -> float:
    """Foreground pixel count."""
    return float(to_numpy_bool(mask).sum())


def perimeter(mask) -> float:
    """Pixel-unit perimeter: count of foreground pixels touching background.

    A foreground pixel is on the perimeter if at least one of its 4-neighbours
    is background (pixels on the image edge count the outside as background).
    """
    m = to_numpy_bool(mask)
    if not m.any():
        return 0.0
    p = np.pad(m, 1, mode="constant", constant_values=False)
    up, down = p[:-2, 1:-1], p[2:, 1:-1]
    left, right = p[1:-1, :-2], p[1:-1, 2:]
    touches_bg = (~up) | (~down) | (~left) | (~right)
    return float((m & touches_bg).sum())


def area_ratio(mask, total_pixels: float | None = None) -> float:
    """Foreground area divided by total pixels (defaults to the mask size)."""
    m = to_numpy_bool(mask)
    denom = float(total_pixels) if total_pixels else float(m.size)
    if denom <= 0:
        return 0.0
    return float(m.sum()) / denom


# Name → function map (area_ratio handled separately because it needs total).
MORPHOLOGY_FNS = {
    "area": area,
    "perimeter": perimeter,
    "area_ratio": area_ratio,
}
