"""Baseline model registry.

Maps ``model.name`` to a concrete baseline builder for ``provider=baseline``.
This is the single dispatch point for baseline models; the factory never
imports a concrete baseline class directly.

TransUNet lives at ``src/models/baselines/transunet/`` (moved in Phase B).
"""

from __future__ import annotations

from torch import nn


def build_baseline_model(model_cfg, dataset_cfg) -> nn.Module:
    """Build a baseline model from ``model_cfg`` (dispatch on ``name``)."""
    name = str(model_cfg.get("name", "")).lower()

    if name == "transunet":
        from src.models.baselines.transunet import build_transunet
        return build_transunet(model_cfg, dataset_cfg)

    raise ValueError(
        f"Unknown baseline model: '{name}'. Available: 'transunet'."
    )
