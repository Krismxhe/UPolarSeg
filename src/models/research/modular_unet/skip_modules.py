"""
Skip-connection modules + registry (Phase 6).

A SkipModule transforms a skip feature before it is fused into the decoder. It
may use the decoder feature and the decoder level index. Contract:

    forward(skip, decoder_feature=None, level: int = 0) -> skip_out

v1 constraints (kept simple for clean ablation):
  - spatial size is preserved;
  - channel count is preserved (C_out == C_skip), so downstream conv channel
    bookkeeping in the decoder stays valid.

Add a new module by subclassing SkipModule and registering it in SKIP_MODULES;
then select it via ``model.skip.name=<name>``.
"""

import torch.nn as nn


class SkipModule(nn.Module):
    """Base skip module (no-op). Subclasses override forward."""

    def forward(self, skip, decoder_feature=None, level: int = 0):
        return skip


class IdentitySkip(SkipModule):
    """Pass-through skip — the baseline / ablation control."""

    def __init__(self, channels_by_level=None, **params):
        super().__init__()

    def forward(self, skip, decoder_feature=None, level: int = 0):
        return skip


# name → class. New skip modules register here (Phase 6 ships only identity).
SKIP_MODULES = {
    "identity": IdentitySkip,
}


def build_skip_module(skip_cfg, channels_by_level):
    """Instantiate the skip module selected by ``skip_cfg.name`` (default identity).

    Args:
        skip_cfg         : the ``model.skip`` config node.
        channels_by_level: skip channel count per decoder level (for modules that
                           need per-level sizing; ignored by IdentitySkip).
    """
    cfg = skip_cfg or {}
    name = str(cfg.get("name", "identity")).lower()
    if name not in SKIP_MODULES:
        raise ValueError(
            f"Unknown skip module '{name}'. Available: {sorted(SKIP_MODULES)}"
        )
    params = dict(cfg.get("params", {}) or {})
    return SKIP_MODULES[name](channels_by_level=channels_by_level, **params)
