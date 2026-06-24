"""Baseline model registry.

Maps ``model.name`` to a concrete baseline builder for ``provider=baseline``.
This is the single dispatch point for baseline models; the factory never
imports a concrete baseline class directly.

Phase A note: TransUNet has **not** moved yet — it still lives at
``src/models/transunet/``. The import below is therefore a *temporary bridge*
to the top-level package. Phase B will relocate the implementation to
``src/models/baselines/transunet/`` and update only this import.
"""

from __future__ import annotations

from torch import nn


def build_baseline_model(model_cfg, dataset_cfg) -> nn.Module:
    """Build a baseline model from ``model_cfg`` (dispatch on ``name``)."""
    name = str(model_cfg.get("name", "")).lower()

    if name == "transunet":
        # TEMPORARY BRIDGE (Phase A): implementation not yet moved.
        from src.models.transunet import build_transunet
        return build_transunet(model_cfg, dataset_cfg)

    raise ValueError(
        f"Unknown baseline model: '{name}'. Available: 'transunet'."
    )
