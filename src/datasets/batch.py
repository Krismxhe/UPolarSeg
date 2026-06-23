"""
Batch unpacking helper.

Bridges the new dict-style batch (image / mask / metadata) with the legacy
2-tuple batch, so the LightningModule does not need to know which one it is
handed. This keeps old checkpoints / callers working while enabling per-case
evaluation and prediction saving downstream.
"""

from __future__ import annotations

from typing import Any


def unpack_batch(batch: Any):
    """Return (images, masks, metadata) from either a dict batch or a legacy tuple batch.

    - dict batch  → (batch["image"], batch["mask"], batch)
    - 2-tuple/list → (images, masks, {})

    The metadata dict is empty for legacy tuple batches.
    """
    if isinstance(batch, dict):
        return batch["image"], batch["mask"], batch
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, masks = batch
        return images, masks, {}
    raise TypeError(f"Unsupported batch type: {type(batch)}")
