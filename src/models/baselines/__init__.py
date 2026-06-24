"""Baseline model layer.

Established, non-SMP baseline architectures (e.g. TransUNet, and future
UNETR / SwinUNETR) live here. They are built through
:func:`src.models.baselines.registry.build_baseline_model`, which is the only
entry point the model factory uses for ``provider=baseline``.

Baseline code must not import research code.
"""
