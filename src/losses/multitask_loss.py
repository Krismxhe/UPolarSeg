"""
MultiTaskLoss aggregator (Phase 9b skeleton).

Frame-level aggregator over task losses. By default ONLY the segmentation loss
is active, so wrapping the existing seg loss reproduces the baseline total
exactly. Boundary loss is disabled by default and, when enabled, requires the
model to emit ``boundary_logits`` and the batch to carry a ``boundary`` target.

NOTE: this aggregator is intentionally NOT yet wired into ``SegModule`` — the
baseline keeps using ``build_loss`` directly. It is provided so a future
boundary-training pipeline can adopt it without touching the seg loss path.

forward(outputs: dict, targets: dict) -> (total, components: dict)
  outputs : the normalized model output dict (must contain "seg_logits").
  targets : {"mask": Tensor, optional "boundary": Tensor}
"""

from __future__ import annotations

import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, seg_loss, boundary_loss=None,
                 seg_weight: float = 1.0, boundary_weight: float = 0.0,
                 boundary_enabled: bool = False):
        super().__init__()
        self.seg_loss = seg_loss
        self.boundary_loss = boundary_loss
        self.seg_weight = float(seg_weight)
        self.boundary_weight = float(boundary_weight)
        self.boundary_enabled = bool(boundary_enabled)

    def forward(self, outputs: dict, targets: dict):
        if "seg_logits" not in outputs:
            raise ValueError("MultiTaskLoss requires 'seg_logits' in model outputs.")

        seg_total, dice, aux = self.seg_loss(outputs["seg_logits"], targets["mask"])
        total = self.seg_weight * seg_total
        components = {"seg_total": seg_total, "dice": dice, "aux": aux}

        if self.boundary_enabled:
            if self.boundary_loss is None:
                raise ValueError("boundary task enabled but no boundary_loss was provided.")
            if "boundary_logits" not in outputs or "boundary" not in targets:
                raise ValueError(
                    "boundary task enabled but outputs['boundary_logits'] or "
                    "targets['boundary'] is missing."
                )
            boundary = self.boundary_loss(outputs["boundary_logits"], targets["boundary"])
            total = total + self.boundary_weight * boundary
            components["boundary"] = boundary

        components["total"] = total
        return total, components
