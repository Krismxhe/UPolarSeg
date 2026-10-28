"""Per-case quantification readouts and diagnostics with original validity rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from src.quantification.polar_readout import PolarReadout


@torch.no_grad()
def compute_quantification_diagnostics(
    readout: PolarReadout,
    logits: Tensor,
    masks: Tensor,
) -> dict:
    """Return raw readouts plus metric-to-(values, validity) diagnostics.

    The result contains target, hard, soft, gt_soft and metrics. No gradients
    are tracked; training calls read_probabilities directly instead.
    Hard errors require readable GT AND prediction.
    Always report hard_invalid_fraction alongside conditional hard MAE.
    No legacy morphology or fixed-width search window is applied.
    """
    soft = readout.read_probabilities(
        logits.float().softmax(dim=1)
    )

    hard = readout.measure_mask(
        logits.argmax(dim=1)
    )

    target = readout.measure_mask(masks)

    gt_p = torch.stack(
        [masks == k for k in range(3)],
        dim=1,
    ).float()

    gt_soft = readout.read_probabilities(gt_p)

    ref = target["reference_valid"]
    pair = ref & hard["reference_valid"]
    all_cases = torch.ones_like(ref)

    stats = {}

    for key, suffix in (
        ("pmw", "_px"),
        ("dv", "_px"),
        ("pvdr", ""),
    ):
        stats[f"{key}_soft_mae{suffix}"] = (
            (soft[key] - target[key]).abs(),
            ref,
        )

        stats[f"{key}_hard_mae{suffix}"] = (
            (hard[key] - target[key]).abs(),
            pair,
        )

        stats[f"{key}_soft_hard_gap{suffix}"] = (
            (soft[key] - hard[key]).abs(),
            pair,
        )

        stats[f"gt_{key}_proxy_gap{suffix}"] = (
            (gt_soft[key] - target[key]).abs(),
            ref,
        )

    stats["reference_valid_fraction"] = (
        ref.float(),
        all_cases,
    )

    stats["aux_valid_fraction"] = (
        target["aux_valid"].float(),
        all_cases,
    )

    stats["hard_invalid_fraction"] = (
        (~hard["reference_valid"]).float(),
        ref,
    )

    stats["soft_dv_floor_fraction"] = (
        (
            soft["dv"]
            < readout.denominator_floor_px
        ).float(),
        ref,
    )

    return {
        "target": target,
        "hard": hard,
        "soft": soft,
        "gt_soft": gt_soft,
        "metrics": stats,
    }
