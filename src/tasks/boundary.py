"""
Boundary task (Phase 9b).

First version is eval-only: boundary metrics are derived from the segmentation
prediction vs target via mask_to_boundary (see src.metrics.boundary_metrics).
Training a dedicated boundary output (BoundaryHead + boundary loss) is left as a
skeleton — boundary loss is disabled by default in MultiTaskLoss.

``make_boundary_target`` is provided so a future boundary-training pipeline can
generate targets from masks.
"""

from __future__ import annotations

from src.metrics.boundary_metrics import boundary_scores  # noqa: F401 (re-export)
from src.tasks import get_output_cfg, is_task_enabled
from src.utils.geometry import mask_to_boundary

NAME = "boundary"


def is_enabled(cfg) -> bool:
    return is_task_enabled(cfg, NAME)


def boundary_width_px(cfg) -> int:
    return int(get_output_cfg(cfg, NAME).get("boundary_width_px", 3))


def make_boundary_target(mask, width_px: int = 3):
    """Binary boundary target from a class mask (H×W bool). Used for future
    boundary-head training; eval metrics call mask_to_boundary directly."""
    return mask_to_boundary(mask, width_px=width_px)
