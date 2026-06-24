"""
Segmentation task (Phase 9b).

The primary, always-on task. Its output key in the model-output dict is
``seg_logits`` (see src.models.outputs). This module exists mainly so the task
layer is symmetric with boundary/clinical; the actual segmentation loss/metrics
live in src.losses and src.metrics.
"""

from __future__ import annotations

from src.tasks import get_output_cfg

NAME = "segmentation"


def seg_logits_key(cfg) -> str:
    return str(get_output_cfg(cfg, NAME).get("key", "seg_logits"))
