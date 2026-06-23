"""
Model factory — single entry point for building segmentation models.

``SegModule`` calls ``build_model(cfg.model, cfg.dataset)`` and stays agnostic
to the concrete provider. New providers (custom TransUNet, ModularUNet, …) are
added here in later phases without touching the LightningModule.

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

    # Custom providers (transunet, modular_unet, …) are introduced in later
    # phases. Fail loudly rather than silently falling back to SMP.
    raise ValueError(
        f"Unknown model provider: '{provider}'. "
        "Only 'smp' is supported so far; custom providers arrive in later phases."
    )
