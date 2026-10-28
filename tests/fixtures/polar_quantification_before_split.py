# FROZEN NUMERICAL REFERENCE — DO NOT EDIT.
# This is the original pre-split polar_quantification.py from the uploaded ZIP.
# It is kept only for original-versus-refactored numerical/gradient regression
# tests; application code must never import this fixture as its implementation.
# The original source is preserved unchanged below this comment header.

from __future__ import annotations

import math
import torch 
import torch.distributed as dist
from torch import Tensor, nn

def _opposite_vertical(
        profile: Tensor,
        angle_deg: float = 90.0,
)-> Tensor:
    """Sum opposite radii on a fixed axis, with periodic interpolation."""

    n = profile.shape[-1]
    values = []

    for angle in (angle_deg, angle_deg + 180.0):
        x = (angle % 360.0) * n / 360.0
        i = math.floor(x)
        t = x - i

        values.append(
            (1.0 - t) * profile[..., i % n] + t * profile[..., (i+1) % n] 
        )
    
    return values[0] + values[1]

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

class PolarQuantificationLoss(nn.Module):
    def __init__(self, 
                 valid_grid: Tensor,
                 max_radius: float,
                 od_class: int = 1,
                 ppa_class: int = 2,
                 denominator_floor_px: float = 1.0,
                 vertical_angle_deg: float = 90.0,
    ):
        super().__init__()

        if valid_grid.ndim != 2 or min(valid_grid.shape) < 4:
            raise ValueError(
                "valid_grid must be [angle, radius], both >= 4."
            )

        if not math.isfinite(max_radius) or max_radius <= 0:
            raise ValueError(
                "max_radius must be finite and positive."
            )

        if (
            not math.isfinite(denominator_floor_px)
            or denominator_floor_px <= 0
        ):
            raise ValueError(
                "denominator_floor_px must be finite and positive."
            )

        if not math.isfinite(vertical_angle_deg):
            raise ValueError(
                "vertical_angle_deg must be finite."
            )

        if {od_class, ppa_class} != {1, 2}:
            raise ValueError(
                "Requires background=0 and OD/PPA=1/2."
            )

        valid = valid_grid.bool()

        if not bool(valid[:, 0].all()):
            raise ValueError(
                "Polar origin must be inside the image."
            )

        if bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError(
                "Image-valid samples must form radial prefixes."
            )
        
        self.od_class = int(od_class)
        self.ppa_class = int(ppa_class)
        self.max_radius = float(max_radius)

        self.denominator_floor_px = float(
            denominator_floor_px
        )

        # A diameter axis is undirected: theta and theta+180 are identical
        self.vertical_angle_deg = (
            float(vertical_angle_deg) % 180
        )

        self.angle_bins, self.radius_bins = valid.shape

        dr = self.max_radius / self.radius_bins

        j = torch.arange(
            self.radius_bins,
            device = valid.device
        )

        weights = valid.float() * dr

        # Last-sample convention
        # OD at indices 0..k has radius K * dr not (k+1) * dr
        weights[:, 0] = 0.0

        last_valid = torch.where(
            valid,
            j,
            -1,
        ).amax(dim=-1)

        # Geometry only: no trainable parameters or persistent weight keys
        for name, value in (
            ("valid_grid", valid),
            ("ray_weights", weights),
            ("radii", j.float() * dr),
            ("last_valid", last_valid),
        ):
            self.register_buffer(
                name,
                value,
                persistent=False,
            )

    def _check_mask(self, masks: Tensor) -> None:
        expected = (
            self.angle_bins,
            self.radius_bins,
        )

        if (
            masks.ndim != 3
            or tuple(masks.shape[-2:]) != expected
        ):
            raise ValueError(
                "Expected masks [B, angle_bins, radius_bins]."
            )

        if (
            masks.is_floating_point()
            or masks.dtype == torch.bool
        ):
            raise TypeError(
                "Masks must contain integer class IDs."
            )

        if bool(((masks < 0) | (masks > 2)).any()):
            raise ValueError(
                "Only labels 0/1/2 are supported; no ignore labels."
            )

    def _summarize(
        self,
        od: Tensor,
        width: Tensor,
    ) -> dict[str, Tensor]:
        pmw = width.amax(dim=-1)

        dv = _opposite_vertical(
            od,
            self.vertical_angle_deg,
        )

        return {
            # 正式标量读出保持不变。
            "pmw": pmw,
            "dv": dv,
            "pvdr": pmw / dv.clamp_min(
                self.denominator_floor_px
            ),

            # 新增：供候选监督使用。
            "od_profile": od,
            "width_profile": width,
        }

    def _candidate_error(
        self,
        soft: dict,
        target: dict,
        k: int | None = 10,
        angle_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """逐角度计算候选误差，再对每个样本求平均。

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

        # 选择角度不反传，但下面参与误差计算的 pred_w 不 detach。
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

    def read_probabilities(
        self,
        probabilities: Tensor,
    ) -> dict[str, Tensor]:
        """Differentiable probability-weighted radial occupation.

        This is a soft surrogate. Do not detach probabilities here.
        """
        if (
            probabilities.ndim != 4 or probabilities.shape[1] != 3
        ):
            raise ValueError(
                "Expected probabilities [B, 3, angle, radius]."
            )

        expected = (
            self.angle_bins,
            self.radius_bins,
        )

        if tuple(probabilities.shape[-2:]) != expected:
            raise ValueError(
                "Prediction grid differs from cfg.polar.geometry."
            )

        p = probabilities.float()
        w = self.ray_weights.float()

        od = (
            p[:, self.od_class] * w
        ).sum(dim=-1)

        width = (
            p[:, self.ppa_class] * w
        ).sum(dim=-1)

        return self._summarize(od, width)

    @torch.no_grad()
    def measure_mask(
        self,
        masks: Tensor,
    ) -> dict[str, Tensor]:
        """Hard radial-boundary measurement with explicit validity flags.

        reference_valid:
            OD is centred and radial-prefix shaped;
            foreground is not clipped or outside the valid grid;
            the vertical diameter is nondegenerate.

        aux_valid:
            additionally requires OD|PPA radial prefixes.
            Only then does one-hot occupancy equal boundary measurement.

        Invalid protected numeric values must not be reported as valid
        measurements.
        """

        self._check_mask(masks)

        od = (
            (masks == self.od_class)
            & self.valid_grid
        )

        op = (
            (
                (masks == self.od_class)
                | (masks == self.ppa_class)
            )
            & self.valid_grid
        )

        r = self.radii.float()

        r_od = torch.where(
            od,
            r,
            0.0,
        ).amax(dim=-1)

        r_op = torch.where(
            op, 
            r,
            0.0,
        ).amax(dim=-1)

        result = self._summarize(
            r_od,
            (r_op - r_od).clamp_min(0.0),
        )

        centred = od[..., 0].all(dim=-1)

        # Foreground re-entry after background means a broken radial prefix
        od_reentry = (
            (~od[..., :-1] & od[..., 1:])
        ).any(dim=(-2, -1))

        op_reentry = (
            (~op[..., :-1]) & op[..., 1:]
        ).any(dim=(-2, -1))

        edge = self.last_valid[
            None, :, None
        ].expand(masks.shape[0], -1, 1)

        clipped = op.gather(
            -1,
            edge,
        ).any(dim=(-2, -1))

        outside = (
            (masks != 0) & ~self.valid_grid
        ).any(dim=(-2, -1))

        # A disconnected OD is not accepted as the intended central chord.
        od_prefix = centred & ~od_reentry

        result["reference_valid"] = (
            od_prefix
            & ~clipped
            & ~outside
            & (
                result["dv"]
                >= self.denominator_floor_px
            )
        )

        result["integral_compatible"] = (
            od_prefix & ~op_reentry
        )

        result["aux_valid"] = (
            result["reference_valid"]
            & result["integral_compatible"]
        )

        return result

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
    def diagnostics(
        self,
        logits: Tensor,
        masks: Tensor,
    ):
        """Return per-case (values, validity) pairs for epoch statistics.

        Hard errors require readable GT AND prediction.
        Always report hard_invalid_fraction alongside conditional hard MAE.
        No legacy morphology or fixed-width search window is applied.
        """
        soft = self.read_probabilities(
            logits.float().softmax(dim=1)
        )

        hard = self.measure_mask(
            logits.argmax(dim=1)
        )

        target = self.measure_mask(masks)

        gt_p = torch.stack(
            [masks == k for k in range(3)],
            dim=1,
        ).float()

        gt_soft = self.read_probabilities(gt_p)

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
                < self.denominator_floor_px
            ).float(),
            ref,
        )

        return stats 

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

    dataset = cfg["dataset"]

    if (
        int(dataset["num_classes"]) != 3
        or dataset["mask_mode"] != "index"
    ):
        raise ValueError(
            "Requires 3-class index masks."
        )

    polar = cfg.get("polar") or {}

    if not polar.get("enabled", False):
        raise ValueError(
            "Quantification requires polar=enabled."
        )

    g = polar.get("geometry") or {}

    if g.get("mode", "linear") != "linear":
        raise ValueError(
            "This readout does not support logarithmic radial grids."
        )


    size = int(cfg["train"]["img_size"])
    center = g.get("center")

    if center is not None:
        center = tuple(float(x) for x in center)

        if (
            len(center) != 2
            or not all(
                math.isfinite(x) and 0 <= x < size
                for x in center
            )
        ):
            raise ValueError(
                "polar.geometry.center must be "
                "(x,y) inside resized ROI."
            )

    axis = float(q.get("vertical_angle_deg", 90.0))

    if not math.isfinite(axis):
        raise ValueError(
            "vertical_angle_deg must be finite."
        )


    midpoint = (
        (size - 1) / 2.0,
        (size - 1) / 2.0,
    )

    # Lazy imports: disabled/old checkpoints do not enter this path.
    import numpy as np
    from src.transforms.polar_transforms import PolarTransform

    warp = PolarTransform(
        radius_bins=int(g.get("radius_bins") or size),
        angle_bins=int(g.get("angle_bins") or size),
        center=center,
        max_radius=g.get("max_radius"),
        pt_type=g.get("pt_type") or "corner",
        mode="linear",
    )

    ones = np.ones(
        (size, size),
        dtype=np.uint8,
    )

    _, _, max_radius = warp._resolve_geometry(ones)

    if (
        not math.isfinite(max_radius)
        or max_radius <= 0
    ):
        raise ValueError(
            "Resolved polar max_radius must be positive and finite."
        )

    valid = torch.from_numpy(
        warp.apply_to_mask(ones).copy()
    ).bool()

    return PolarQuantificationLoss(
        valid_grid=valid,
        max_radius=max_radius,
        od_class=int(q.get("od_class", 1)),
        ppa_class=int(q.get("ppa_class", 2)),
        denominator_floor_px=float(
            q.get("denominator_floor_px", 1.0)
        ),
        vertical_angle_deg=axis,
    )


