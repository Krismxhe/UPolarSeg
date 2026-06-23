"""
Unified model output contract.

All segmentation models in this codebase are consumed through a single, stable
output schema so that ``SegModule`` does not need to know whether a model
returns a bare logits tensor (current SMP / TransUNet baselines) or a richer
dict (future multi-task models with boundary / clinical / feature outputs).

Canonical schema (dict)::

    {
        "seg_logits": Tensor,   # REQUIRED — B×C×H×W segmentation logits
        # optional, introduced in later phases:
        # "boundary_logits": Tensor,
        # "clinical": ...,
        # "features": ...,
    }

``normalize_model_output`` is the only adapter: it wraps a bare tensor into the
canonical dict and validates dict outputs. It performs NO activation
(no sigmoid / softmax) and never changes the tensor's dtype, device or shape.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

# Canonical key for the primary segmentation logits.
SEG_LOGITS_KEY = "seg_logits"


def normalize_model_output(model_output: Any) -> Dict[str, Any]:
    """Coerce a model's forward output into the canonical output dict.

    Args:
        model_output: either a ``torch.Tensor`` of segmentation logits, or a
            dict that already follows the canonical schema.

    Returns:
        A dict guaranteed to contain ``"seg_logits"``.

    Raises:
        ValueError: if a dict is given but lacks ``"seg_logits"``.
        TypeError:  if the output is neither a Tensor nor a dict.
    """
    if isinstance(model_output, torch.Tensor):
        # Wrap as-is: no activation, no dtype/device/shape change.
        return {SEG_LOGITS_KEY: model_output}

    if isinstance(model_output, dict):
        if SEG_LOGITS_KEY not in model_output:
            raise ValueError(
                f"Model output dict is missing the required '{SEG_LOGITS_KEY}' key; "
                f"got keys: {sorted(model_output.keys())}"
            )
        return model_output

    raise TypeError(
        "Model output must be a torch.Tensor or a dict containing "
        f"'{SEG_LOGITS_KEY}', got {type(model_output)}"
    )
