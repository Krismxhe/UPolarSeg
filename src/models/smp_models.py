"""
SMP (segmentation-models-pytorch) model builder.

Thin wrapper around ``smp.create_model`` so the rest of the codebase can build
SMP architectures through the unified model factory without importing smp
directly.
"""

import segmentation_models_pytorch as smp
from omegaconf import OmegaConf


def _as_dict(value):
    """Return a plain dict for an OmegaConf node / dict / None."""
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def build_smp_model(model_cfg, dataset_cfg):
    """Build an SMP model.

    ``classes`` always comes from the dataset config (single source of truth);
    ``params`` allows passing extra arch-specific kwargs without code changes.
    """
    return smp.create_model(
        arch=model_cfg.arch,
        encoder_name=model_cfg.encoder,
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=model_cfg.get("in_channels", 3),
        classes=dataset_cfg.num_classes,
        **_as_dict(model_cfg.get("params", None)),
    )
