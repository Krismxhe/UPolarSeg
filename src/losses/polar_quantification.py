"""
Compatibility loss API built on independent polar measurement readouts.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor

from src.quantification.diagnostics import compute_quantification_diagnostics
from src.quantification.polar_readout import PolarReadout, _build_polar_readout
from src.quantification.polar_readout import _opposite_vertical as _opposite_vertical


def _valid_mean(value: Tensor, valid: Tensor) -> Tensor:
    """Global valid-case gradient mean for ordinary DDP gradient averaging.

    All ranks must call this function.
    All-invalid batches return a graph-connected zero.
    Average the returned local loss across ranks when logging.
    """
    numerator = torch.where(
        valid,
        value,
        torch.zeros_like(value),
    ).sum()

    count = valid.sum().to(value.dtype)
    world = 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        world = dist.get_world_size()

    return numerator * world / count.clamp_min(1.0)


class PolarQuantificationLoss(PolarReadout):
    """Original candidate-angle objective with inherited readout compatibility."""

    def _candidate_error(
        self,
        soft: dict,
        target: dict,
        k: int | None = 10,
        angle_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        逐角度计算候选误差，再对每个样本求平均。

        k > 0:
            选择允许范围内的：
            预测 Top-k 与正宽度 GT Top-k 的角度并集。

        k is None:
            监督允许范围内的全部角度。

        angle_mask:
            None、[angle] 或 [B, angle] 的布尔掩膜。

        Returns:
            error: [B]，每个样本的候选误差。
            has_candidates: [B]，是否至少选中一个角度。
        """
        pred_w = soft["width_profile"]
        true_w = target["width_profile"].detach()

        if (
            pred_w.ndim != 2
            or pred_w.shape != true_w.shape
        ):
            raise ValueError(
                "Profiles must have matching [B, angle] shapes."
            )

        if k is not None and (
            type(k) is not int or k <= 0
        ):
            raise ValueError(
                "k must be a positive integer or None."
            )

        if angle_mask is None:
            allowed = torch.ones_like(
                pred_w,
                dtype=torch.bool,
            )
        else:
            allowed = angle_mask.to(
                device=pred_w.device,
                dtype=torch.bool,
            )

            if (
                allowed.ndim == 1
                and allowed.shape[0] == pred_w.shape[1]
            ):
                allowed = allowed.unsqueeze(0).expand_as(
                    pred_w
                )

            if allowed.shape != pred_w.shape:
                raise ValueError(
                    "angle_mask must be [angle] or [B, angle]."
                )

        # 选择角度不反传，但下面参与误差计算的 pred_w 和 true_w 仍然保留梯度。
        with torch.no_grad():
            if k is None:
                selected = allowed
            else:
                count = min(k, pred_w.shape[-1])

                pred_idx = (
                    pred_w.detach()
                    .masked_fill(~allowed, -torch.inf)
                    .topk(
                        count,
                        dim=-1,
                        sorted=False,
                    )
                    .indices
                )

                true_idx = (
                    true_w
                    .masked_fill(~allowed, -torch.inf)
                    .topk(
                        count,
                        dim=-1,
                        sorted=False,
                    )
                    .indices
                )

                pred_selected = torch.zeros_like(
                    allowed
                ).scatter_(-1, pred_idx, True)

                true_selected = torch.zeros_like(
                    allowed
                ).scatter_(-1, true_idx, True)

                # GT 的零宽度方向不额外加入。
                # 预测选中的零 GT 方向仍保留，用于抑制假阳性。
                selected = (
                    pred_selected
                    | (true_selected & (true_w > 0))
                ) & allowed

        pred_v = pred_w / soft["dv"].clamp_min(
            self.denominator_floor_px
        )[:, None]

        true_v = true_w / target["dv"].detach().clamp_min(
            self.denominator_floor_px
        )[:, None]

        # 同一角度逐项匹配，不按排序名次配对。
        per_angle = (
            (pred_w - true_w).abs() / self.max_radius
            + (pred_v - true_v).abs()
        )

        count = selected.sum(dim=-1)

        error = torch.where(
            selected,
            per_angle,
            torch.zeros_like(per_angle),
        ).sum(dim=-1) / count.clamp_min(1)

        return error, count > 0

    def forward(
        self,
        logits: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, dict]:
        if logits.shape[0] != masks.shape[0]:
            raise ValueError(
                "Logits and masks have different batch sizes."
            )

        with torch.autocast(
            device_type=logits.device.type,
            enabled=False,
        ):
            soft = self.read_probabilities(
                logits.float().softmax(dim=1)
            )

            target = self.measure_mask(masks)

            error, has_candidates = self._candidate_error(
                soft,
                target,
                k=10,
                angle_mask=None,
            )

            valid = (
                target["aux_valid"]
                & has_candidates
            )

            loss = _valid_mean(error, valid)

        return loss, {
            "aux_valid_fraction": (
                valid.float().mean().detach()
            )
        }

    @torch.no_grad()
    def diagnostics(self, logits: Tensor, masks: Tensor):
        """Return the original metric-to-(values, validity) mapping."""
        return compute_quantification_diagnostics(self, logits, masks)["metrics"]


QUANT_VAL_KEYS = tuple(
    name
    for key, suffix in (
        ("pmw", "_px"),
        ("dv", "_px"),
        ("pvdr", ""),
    )
    for name in (
        f"{key}_soft_mae{suffix}",
        f"{key}_hard_mae{suffix}",
        f"{key}_soft_hard_gap{suffix}",
        f"gt_{key}_proxy_gap{suffix}",
    )
) + (
    "reference_valid_fraction",
    "aux_valid_fraction",
    "hard_invalid_fraction",
    "soft_dv_floor_fraction",
)


def build_polar_quantification(
    cfg,
) -> PolarQuantificationLoss | None:
    """Reuse actual repository geometry; old configs remain disabled."""
    q = cfg.get("quantification") or {}

    if not q.get("enabled", False):
        return None

    weight = float(q.get("weight", 0.1))

    if not math.isfinite(weight) or weight < 0:
        raise ValueError(
            "quantification.weight must be finite and nonnegative."
        )

    return _build_polar_readout(cfg, PolarQuantificationLoss)


