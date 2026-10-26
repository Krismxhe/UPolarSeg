from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from monai.metrics import compute_dice, compute_hausdorff_distance, compute_surface_dice
except Exception as exc:  # pragma: no cover - explicit runtime error in _require_monai()
    torch = None
    compute_dice = compute_hausdorff_distance = compute_surface_dice = None
    _MONAI_IMPORT_ERROR = exc
else:
    _MONAI_IMPORT_ERROR = None


# v2: Dice/NSD/HD95 come from MONAI, whose boundary uses 4-neighbour erosion (v1 used
# all 8 neighbours), so v2 NSD/HD95 values are not directly comparable with v1 records.
PROTOCOL_VERSION = "odppa-v2"


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj.get(key, default)
    except AttributeError:
        return getattr(obj, key, default)


def odppa_cfg(cfg):
    eval_cfg = _cfg_get(cfg, "eval", {}) or {}
    return _cfg_get(eval_cfg, "odppa", {}) or {}


def enabled(cfg) -> bool:
    return bool(_cfg_get(odppa_cfg(cfg), "enabled", False))


def _as_hw(img_size) -> Tuple[int, int]:
    if isinstance(img_size, (list, tuple)):
        if len(img_size) != 2:
            raise ValueError(f"train.img_size must be int or [H,W], got {img_size!r}")
        return int(img_size[0]), int(img_size[1])
    x = int(img_size)
    return x, x


def _resolve_polar_geometry(cfg) -> Tuple[int, int, Tuple[float, float], float, str]:
    """Geometry of the Cartesian grid immediately before PolarTransform."""
    cart_h, cart_w = _as_hw(cfg.train.img_size)
    geometry = cfg.polar.geometry

    center_cfg = _cfg_get(geometry, "center", None)
    if center_cfg is None:
        center = ((cart_w - 1) / 2.0, (cart_h - 1) / 2.0)
    else:
        center = (float(center_cfg[0]), float(center_cfg[1]))

    max_radius = _cfg_get(geometry, "max_radius", None)
    if max_radius is None:
        pt_type = str(_cfg_get(geometry, "pt_type", "corner")).lower()
        cx, cy = center
        if pt_type == "corner":
            corners = np.asarray(
                [[0, 0], [cart_w - 1, 0], [0, cart_h - 1], [cart_w - 1, cart_h - 1]],
                dtype=np.float64,
            )
            max_radius = float(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy).max())
        elif pt_type == "min_edge":
            max_radius = float(min(cx, cy, (cart_w - 1) - cx, (cart_h - 1) - cy))
        else:
            raise ValueError(f"Unsupported polar.geometry.pt_type={pt_type!r}")
    mode = str(_cfg_get(geometry, "mode", "linear")).lower()
    if mode not in {"linear", "log"}:
        raise ValueError(f"Unsupported polar.geometry.mode={mode!r}")
    return cart_h, cart_w, center, float(max_radius), mode


def inverse_polar_hard_mask(mask_polar: np.ndarray, cfg) -> np.ndarray:
    cart_h, cart_w, center, max_radius, mode = _resolve_polar_geometry(cfg)
    polar_flag = cv2.WARP_POLAR_LOG if mode == "log" else cv2.WARP_POLAR_LINEAR
    flags = polar_flag | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST
    cart = cv2.warpPolar(
        mask_polar.astype(np.float32, copy=False),
        dsize=(int(cart_w), int(cart_h)),
        center=(float(center[0]), float(center[1])),
        maxRadius=float(max_radius),
        flags=flags,
    )
    cart = np.nan_to_num(cart, nan=0.0)
    return np.rint(cart).astype(np.int64)


def restore_prediction_to_native(pred_native: np.ndarray, gt_shape: Tuple[int, int], cfg) -> np.ndarray:
    pred_native = np.asarray(pred_native, dtype=np.int64)
    if bool(cfg.polar.enabled):
        pred_cart = inverse_polar_hard_mask(pred_native, cfg)
    else:
        pred_cart = pred_native
    gt_h, gt_w = int(gt_shape[0]), int(gt_shape[1])
    if pred_cart.shape != (gt_h, gt_w):
        pred_cart = cv2.resize(
            pred_cart.astype(np.int32), (gt_w, gt_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int64)
    return pred_cart


def load_native_mask(mask_path: str, mask_mode: str = "index") -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        # Index masks are expected to be single-channel. If a PNG is RGB but all
        # channels are equal, accepting channel 0 is safe; otherwise fail loudly.
        if not np.all(mask[..., 0] == mask[..., 1]) or not np.all(mask[..., 0] == mask[..., 2]):
            raise ValueError(f"Expected an index mask, got non-identical RGB channels: {mask_path}")
        mask = mask[..., 0]
    if mask_mode == "binary":
        mask = (mask > 127).astype(np.int64)
    else:
        mask = mask.astype(np.int64, copy=False)
    return mask


def _require_monai() -> None:
    if _MONAI_IMPORT_ERROR is not None:
        raise ImportError(
            "OD/PPA Dice/NSD/HD95 require MONAI. Install it with `pip install monai`."
        ) from _MONAI_IMPORT_ERROR


def _as_bchw(mask: np.ndarray) -> torch.Tensor:
    """HxW binary mask -> [1, 1, H, W] bool tensor, MONAI's (batch, one-hot class) layout."""
    return torch.tensor(np.asarray(mask, dtype=bool))[None, None]


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice. NaN for an empty reference, 0 for an empty prediction (MONAI ``ignore_empty``)."""
    _require_monai()
    dsc = compute_dice(
        _as_bchw(pred), _as_bchw(target), include_background=True, ignore_empty=True
    )
    return float(dsc)


def nsd_score(pred: np.ndarray, target: np.ndarray, tolerance_px: float) -> float:
    """Normalised surface Dice at ``tolerance_px`` pixels.

    NaN for an empty reference, 0 for an empty prediction, and NaN when the
    tolerance itself is undefined (e.g. derived from a missing OD reference).
    """
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(target, dtype=bool)
    if not g.any():
        return float("nan")
    if not p.any():
        return 0.0
    tol = float(tolerance_px)
    if not math.isfinite(tol):
        return float("nan")
    _require_monai()
    nsd = compute_surface_dice(
        _as_bchw(p), _as_bchw(g), class_thresholds=[tol], include_background=True
    )
    return float(nsd)


def hd95_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels; NaN if either mask is empty."""
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(target, dtype=bool)
    if not g.any() or not p.any():
        return float("nan")
    _require_monai()
    hd = compute_hausdorff_distance(
        _as_bchw(p), _as_bchw(g), include_background=True, percentile=95
    )
    return float(hd)


def equivalent_diameter(area_px: int) -> float:
    if area_px <= 0:
        return float("nan")
    return float(2.0 * math.sqrt(float(area_px) / math.pi))


def load_metadata_csv(path: Optional[str]) -> Dict[str, dict]:
    """Optional case metadata keyed by case_id.

    Supported columns: case_id, patient_id, eye, reference_complete.
    Missing patient_id falls back to case_id. Missing reference_complete defaults True.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        try:
            from hydra.utils import to_absolute_path
            p = Path(to_absolute_path(str(p)))
        except Exception:
            p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"eval.odppa.metadata_csv not found: {p}")
    out: Dict[str, dict] = {}
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                raise ValueError(f"metadata_csv row missing case_id: {row}")
            if case_id in out:
                raise ValueError(f"Duplicate case_id in metadata_csv: {case_id}")
            out[case_id] = row
    return out


def _parse_bool(value, default=True) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def build_case_record(
    *,
    pred_native: np.ndarray,
    mask_path: str,
    case_id: str,
    image_path: str,
    cfg,
    metadata: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build one scalar-only OD/PPA evaluation record."""
    ecfg = odppa_cfg(cfg)
    od_id = int(_cfg_get(ecfg, "od_class_id", 1))
    ppa_id = int(_cfg_get(ecfg, "ppa_class_id", 2))
    nsd_ratio = float(_cfg_get(ecfg, "nsd_tolerance_od_ratio", 0.01))
    nsd_px_override = _cfg_get(ecfg, "nsd_tolerance_px", None)

    gt = load_native_mask(mask_path, mask_mode=str(cfg.dataset.mask_mode))
    pred = restore_prediction_to_native(pred_native, gt.shape[:2], cfg)

    valid_ids = set(np.unique(gt).tolist())
    expected = set(range(int(cfg.dataset.num_classes)))
    if not valid_ids.issubset(expected):
        raise ValueError(
            f"GT mask {mask_path} has labels {sorted(valid_ids)} outside {sorted(expected)}"
        )

    md = dict(metadata or {})
    patient_id = str(md.get("patient_id") or case_id)
    eye = str(md.get("eye") or "")
    reference_complete = _parse_bool(md.get("reference_complete"), default=True)

    gt_od = gt == od_id
    gt_ppa = gt == ppa_id
    pr_od = pred == od_id
    pr_ppa = pred == ppa_id
    od_area = int(gt_od.sum())
    ppa_area = int(gt_ppa.sum())
    pr_od_area = int(pr_od.sum())
    pr_ppa_area = int(pr_ppa.sum())
    od_diam = equivalent_diameter(od_area)
    tol = float(nsd_px_override) if nsd_px_override is not None else float(nsd_ratio * od_diam)

    rec = {
        "protocol_version": PROTOCOL_VERSION,
        "case_id": str(case_id),
        "patient_id": patient_id,
        "eye": eye,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "reference_complete": bool(reference_complete),
        "eval_height": int(gt.shape[0]),
        "eval_width": int(gt.shape[1]),
        "gt_od_area_px": od_area,
        "pred_od_area_px": pr_od_area,
        "gt_ppa_area_px": ppa_area,
        "pred_ppa_area_px": pr_ppa_area,
        "gt_od_equiv_diameter_px": od_diam,
        "nsd_tolerance_px": tol,
        "od_pred_empty": bool(pr_od_area == 0),
        "ppa_pred_empty": bool(pr_ppa_area == 0),
        "od_reference_valid": bool(od_area > 0),
    }

    if od_area > 0:
        rec["od_dsc"] = dice_score(pr_od, gt_od)
        rec["od_nsd"] = nsd_score(pr_od, gt_od, tol)
        if pr_od_area > 0:
            hd = hd95_score(pr_od, gt_od)
            rec["od_hd95_px"] = hd
            rec["od_hd95_norm"] = hd / od_diam
            rec["od_hd95_valid"] = True
        else:
            rec["od_hd95_px"] = float("nan")
            rec["od_hd95_norm"] = float("nan")
            rec["od_hd95_valid"] = False
    else:
        rec.update({
            "od_dsc": float("nan"), "od_nsd": float("nan"),
            "od_hd95_px": float("nan"), "od_hd95_norm": float("nan"),
            "od_hd95_valid": False,
        })

    if not reference_complete:
        ppa_status = "unknown"
    else:
        ppa_status = "positive" if ppa_area > 0 else "negative"
    rec["ppa_status"] = ppa_status

    if ppa_status == "positive":
        rec["ppa_dsc"] = dice_score(pr_ppa, gt_ppa)
        rec["ppa_nsd"] = nsd_score(pr_ppa, gt_ppa, tol)
        if pr_ppa_area > 0:
            hd = hd95_score(pr_ppa, gt_ppa)
            rec["ppa_hd95_px"] = hd
            rec["ppa_hd95_norm"] = hd / od_diam if od_area > 0 else float("nan")
            rec["ppa_hd95_valid"] = bool(od_area > 0)
        else:
            rec["ppa_hd95_px"] = float("nan")
            rec["ppa_hd95_norm"] = float("nan")
            rec["ppa_hd95_valid"] = False
        rec["ppa_negative_correct"] = float("nan")
        rec["ppa_fp_norm"] = float("nan")
    elif ppa_status == "negative":
        rec["ppa_dsc"] = float("nan")
        rec["ppa_nsd"] = float("nan")
        rec["ppa_hd95_px"] = float("nan")
        rec["ppa_hd95_norm"] = float("nan")
        rec["ppa_hd95_valid"] = False
        rec["ppa_negative_correct"] = float(pr_ppa_area == 0)
        rec["ppa_fp_norm"] = (float(pr_ppa_area) / float(od_area)) if od_area > 0 else float("nan")
    else:
        rec["ppa_dsc"] = float("nan")
        rec["ppa_nsd"] = float("nan")
        rec["ppa_hd95_px"] = float("nan")
        rec["ppa_hd95_norm"] = float("nan")
        rec["ppa_hd95_valid"] = False
        rec["ppa_negative_correct"] = float("nan")
        rec["ppa_fp_norm"] = float("nan")

    return rec
