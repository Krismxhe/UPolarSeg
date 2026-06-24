"""
Geometry helpers for boundary / morphology metrics.

Pure NumPy (no scipy/cv2 dependency) so it runs in any environment. All
operations work in pixel units; no physical-unit conversion is performed here.
"""

from __future__ import annotations

import numpy as np


def to_numpy_bool(mask) -> np.ndarray:
    """Convert a torch tensor / numpy array mask to a 2-D boolean array."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask) != 0


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary erosion with a 4-connected structuring element (NumPy only)."""
    m = mask
    for _ in range(max(int(iterations), 0)):
        p = np.pad(m, 1, mode="constant", constant_values=False)
        m = (
            p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        )
    return m


def mask_to_boundary(mask, width_px: int = 3) -> np.ndarray:
    """Return the boundary band of a binary mask as a boolean array.

    boundary = mask AND NOT erode(mask, width_px), i.e. a band ``width_px`` pixels
    thick along the inner edge of the foreground region.
    """
    m = to_numpy_bool(mask)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    width_px = max(int(width_px), 1)
    eroded = _erode(m, width_px)
    return m & ~eroded
