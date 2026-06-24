"""Research model registry.

Maps ``model.name`` to a concrete research builder for ``provider=research``.
This is the single dispatch point for research models; the factory never
imports a concrete research class directly.

Phase A note: ModularUNet has **not** moved yet — it still lives at
``src/models/modular_unet/``. The import below is therefore a *temporary
bridge* to the top-level package. Phase C will relocate the implementation to
``src/models/research/modular_unet/`` and update only this import.
"""

from __future__ import annotations

from torch import nn


def build_research_model(model_cfg, dataset_cfg) -> nn.Module:
    """Build a research model from ``model_cfg`` (dispatch on ``name``)."""
    name = str(model_cfg.get("name", "")).lower()

    if name == "modular_unet":
        # TEMPORARY BRIDGE (Phase A): implementation not yet moved.
        from src.models.modular_unet import build_modular_unet
        return build_modular_unet(model_cfg, dataset_cfg)

    raise ValueError(
        f"Unknown research model: '{name}'. Available: 'modular_unet'."
    )
