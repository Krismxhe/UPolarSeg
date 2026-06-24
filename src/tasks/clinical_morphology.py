"""
Clinical morphology task (Phase 9b).

First version is eval-only and deterministic: morphology metrics (area,
perimeter, area_ratio) are computed from the predicted mask in PIXEL units
(see src.metrics.clinical_metrics). No physical-unit conversion is performed.
"""

from __future__ import annotations

from typing import List

from src.metrics.clinical_metrics import DEFAULT_METRICS, compute_class_clinical  # noqa: F401
from src.tasks import get_output_cfg, is_task_enabled

NAME = "clinical"


def is_enabled(cfg) -> bool:
    return is_task_enabled(cfg, NAME)


def metric_names(cfg) -> List[str]:
    metrics = get_output_cfg(cfg, NAME).get("metrics", None)
    return list(metrics) if metrics else list(DEFAULT_METRICS)
