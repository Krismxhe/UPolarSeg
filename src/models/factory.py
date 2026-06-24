"""
Model factory — single entry point for building segmentation models.

``SegModule`` calls ``build_model(cfg.model, cfg.dataset)`` and stays agnostic
to the concrete provider. ``factory.py`` is the only provider router; it never
lets the LightningModule import a concrete model class.

Provider layers:

- ``smp``      → ``segmentation_models_pytorch`` baselines.
- ``baseline`` → established non-SMP baselines (TransUNet, …) via the
  baselines registry.
- ``research`` → user-owned research models (ModularUNet, …) via the research
  registry.
- ``custom``   → **deprecated** compatibility alias. It maps legacy
  ``custom/<name>`` configs onto the baseline/research registries so old
  commands keep working. Prefer ``baseline`` / ``research`` in new configs.

Backward compatibility: legacy model configs have no ``provider`` field; they
are treated as ``provider: smp``.
"""

from torch import nn


def build_model(model_cfg, dataset_cfg) -> nn.Module:
    """Return a ``torch.nn.Module`` whose forward yields logits B×C×H×W.

    Args:
        model_cfg   : the ``cfg.model`` node (Hydra/OmegaConf).
        dataset_cfg : the ``cfg.dataset`` node (provides ``num_classes``).
    """
    provider = str(model_cfg.get("provider", "smp")).lower()

    if provider == "smp":
        from src.models.smp_models import build_smp_model
        return build_smp_model(model_cfg, dataset_cfg)

    if provider == "baseline":
        from src.models.baselines.registry import build_baseline_model
        return build_baseline_model(model_cfg, dataset_cfg)

    if provider == "research":
        from src.models.research.registry import build_research_model
        return build_research_model(model_cfg, dataset_cfg)

    if provider == "custom":
        return _build_custom_compat_model(model_cfg, dataset_cfg)

    raise ValueError(
        f"Unknown model provider: '{provider}'. "
        "Available: 'smp', 'baseline', 'research', 'custom'."
    )


def _build_custom_compat_model(model_cfg, dataset_cfg) -> nn.Module:
    """Deprecated ``provider=custom`` alias.

    Routes legacy ``custom/<name>`` configs to the baseline/research registries
    so existing commands keep working. New configs should use
    ``provider=baseline`` or ``provider=research`` directly.
    """
    name = str(model_cfg.get("name", "")).lower()

    if name == "transunet":
        # custom/transunet → baseline/transunet
        from src.models.baselines.registry import build_baseline_model
        return build_baseline_model(model_cfg, dataset_cfg)

    if name == "modular_unet":
        # custom/modular_unet → research/modular_unet
        from src.models.research.registry import build_research_model
        return build_research_model(model_cfg, dataset_cfg)

    raise ValueError(
        "provider=custom is deprecated and this name could not be mapped: "
        f"'{name}'. Use provider=baseline (transunet) or provider=research "
        "(modular_unet)."
    )
