"""Save class-index predictions in polar and/or Cartesian coordinates.

The writer has no training-framework dependencies. Predictions are hard masks,
not probabilities or display masks: binary labels stay 0/1, and multiclass labels
are stored unchanged in 8-bit or 16-bit PNGs. All resampling is nearest-neighbour.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image


def _cfg_get(node, key, default=None):
    """Read a dict, OmegaConf node, or a simple attribute-based configuration."""
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def normalize_save_mode(value) -> str:
    """Normalize none/polar/cartesian/both, including legacy boolean flags."""
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "both" if value else "none"
    if isinstance(value, str):
        value = value.strip().lower()
        value = {"false": "none", "true": "both", "cart": "cartesian"}.get(value, value)
        if value in {"none", "polar", "cartesian", "both"}:
            return value
    raise ValueError(
        "eval.save_predictions must be 'none', 'polar', 'cartesian', or 'both' "
        f"(legacy false/true are also supported), got {value!r}"
    )


def _positive_int(value, name: str) -> int:
    try:
        result = int(value)
        valid = not isinstance(value, (bool, np.bool_)) and float(value) == result and result > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _as_hw(value, name: str, *, allow_square: bool = True) -> tuple[int, int]:
    if allow_square and np.isscalar(value):
        side = _positive_int(value, name)
        return side, side
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an explicit (H, W) pair, got {value!r}")
    try:
        pair = tuple(value)
    except TypeError:
        pair = ()
    if len(pair) != 2:
        raise ValueError(f"{name} must be an explicit (H, W) pair, got {value!r}")
    return _positive_int(pair[0], f"{name}[0]"), _positive_int(pair[1], f"{name}[1]")


def _input_shape(cfg) -> tuple[int, int]:
    return _as_hw(_cfg_get(_cfg_get(cfg, "train", {}), "img_size"), "train.img_size")


def _resolve_geometry(cfg):
    """Use the same pre-polar Cartesian grid and radius rules as evaluation.

    Missing geometry (e.g. polar=disabled) uses the enabled preset's defaults:
    image-centred, corner radius, linear coordinates, and input-sized bins.
    """
    cart_h, cart_w = _input_shape(cfg)
    geometry = _cfg_get(_cfg_get(cfg, "polar", {}), "geometry", {}) or {}
    center_value = _cfg_get(geometry, "center")
    if center_value is None:
        center = ((cart_w - 1) / 2.0, (cart_h - 1) / 2.0)
    else:
        try:
            center = () if isinstance(center_value, (str, bytes)) else tuple(float(x) for x in center_value)
        except (TypeError, ValueError, OverflowError):
            center = ()
        if len(center) != 2 or not all(math.isfinite(x) for x in center):
            raise ValueError("polar.geometry.center must be a finite (x, y) pair")

    max_radius = _cfg_get(geometry, "max_radius")
    if max_radius is None:
        cx, cy = center
        pt_type = str(_cfg_get(geometry, "pt_type", "corner")).lower()
        if pt_type == "corner":
            max_radius = max(
                math.hypot(x - cx, y - cy)
                for x, y in ((0, 0), (cart_w - 1, 0), (0, cart_h - 1), (cart_w - 1, cart_h - 1))
            )
        elif pt_type == "min_edge":
            max_radius = min(cx, cy, cart_w - 1 - cx, cart_h - 1 - cy)
        else:
            raise ValueError(f"Unsupported polar.geometry.pt_type={pt_type!r}")
    try:
        max_radius = float(max_radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("polar.geometry.max_radius must be finite and positive") from exc
    if not math.isfinite(max_radius) or max_radius <= 0:
        raise ValueError("polar.geometry.max_radius must be finite and positive")

    mode = str(_cfg_get(geometry, "mode", "linear")).lower()
    if mode not in {"linear", "log"}:
        raise ValueError(f"Unsupported polar.geometry.mode={mode!r}")
    if mode == "log" and max_radius <= 1:
        raise ValueError("Log-polar conversion requires polar.geometry.max_radius > 1")

    radius_bins = _cfg_get(geometry, "radius_bins")
    angle_bins = _cfg_get(geometry, "angle_bins")
    polar_shape = (
        _positive_int(cart_h if angle_bins is None else angle_bins, "polar.geometry.angle_bins"),
        _positive_int(cart_w if radius_bins is None else radius_bins, "polar.geometry.radius_bins"),
    )
    return (cart_h, cart_w), polar_shape, center, max_radius, mode


def _label_mask(mask) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"Prediction mask must be a nonempty 2-D class-index array, got shape {values.shape}")
    if values.dtype.kind not in "buif":
        raise ValueError(f"Prediction mask must contain integer class labels, got dtype {values.dtype}")
    if values.dtype.kind == "f":
        if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
            raise ValueError("Prediction mask must contain finite integer class labels, not probabilities")
    minimum, maximum = int(values.min()), int(values.max())
    if minimum < 0 or maximum > 65535:
        raise ValueError("PNG class labels must be in the range 0..65535")
    dtype = np.uint8 if maximum <= 255 else np.uint16
    return np.ascontiguousarray(values, dtype=dtype)


def _case_stem(case_id) -> str:
    if case_id is None:
        raise ValueError("case_id must be a nonempty filename stem")
    stem = str(case_id)
    if (
        not stem.strip()
        or stem in {".", ".."}
        or any(char in stem for char in '/\\<>:"|?*')
        or any(ord(char) < 32 or ord(char) == 127 for char in stem)
    ):
        raise ValueError(f"case_id must be a safe single filename stem, got {case_id!r}")
    return stem


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _warp_mask(mask: np.ndarray, cfg, *, inverse: bool) -> np.ndarray:
    cart_shape, polar_shape, center, max_radius, mode = _resolve_geometry(cfg)
    if not inverse:
        mask = _resize_mask(mask, cart_shape)
    output_shape = cart_shape if inverse else polar_shape
    flags = cv2.INTER_NEAREST | cv2.WARP_FILL_OUTLIERS
    flags |= cv2.WARP_POLAR_LOG if mode == "log" else cv2.WARP_POLAR_LINEAR
    if inverse:
        flags |= cv2.WARP_INVERSE_MAP
    converted = cv2.warpPolar(
        mask.astype(np.float32, copy=False),
        dsize=(output_shape[1], output_shape[0]),
        center=center,
        maxRadius=max_radius,
        flags=flags,
    )
    return _label_mask(converted)


def _atomic_save_png(mask: np.ndarray, destination: Path) -> None:
    """Make each PNG replacement atomic, including repeated DDP sample IDs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            Image.fromarray(mask).save(stream, format="PNG")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class PredictionMaskWriter:
    """Write selected coordinate representations of one prediction at a time.

    ``original_size`` is always (H, W), never (W, H). ``cartesian_size='input'``
    writes on ``train.img_size``; ``'original'`` restores the inverse-projected
    input grid to the original image dimensions with nearest-neighbour resizing.

    For a Cartesian model, polar output is a forward projection of its predicted
    Cartesian mask. For a polar model, the polar prediction is saved unchanged.
    All geometry uses the resized input grid, including an explicitly set centre.
    """

    def __init__(
        self, cfg, output_dir, mode=None, cartesian_size=None, *,
        polar_dir=None, cartesian_dir=None,
    ):
        self.cfg = cfg
        eval_cfg = _cfg_get(cfg, "eval", {}) or {}
        self.mode = normalize_save_mode(
            _cfg_get(eval_cfg, "save_predictions", "none") if mode is None else mode
        )
        if cartesian_size is None:
            cartesian_size = _cfg_get(eval_cfg, "prediction_cartesian_size", "original")
        self.cartesian_size = str(cartesian_size).strip().lower()
        if self.cartesian_size not in {"original", "input"}:
            raise ValueError("eval.prediction_cartesian_size must be 'original' or 'input'")
        self.output_dir = Path(output_dir)
        self.polar_dir = Path(polar_dir) if polar_dir is not None else self.output_dir / "masks_polar"
        self.cartesian_dir = (
            Path(cartesian_dir) if cartesian_dir is not None else self.output_dir / "masks_cartesian"
        )
        self.input_is_polar = bool(_cfg_get(_cfg_get(cfg, "polar", {}), "enabled", False))

    def save(self, mask, case_id, original_size=None) -> dict[str, str]:
        """Save a 2-D class-index mask and return paths (empty if not requested).

        ``pred_path`` prefers the Cartesian result when present, otherwise polar.
        Repeated IDs atomically replace existing files. Both outputs are prepared
        before writing, so invalid input/configuration does not leave half a pair.
        """
        paths = {"pred_path": "", "pred_polar_path": "", "pred_cartesian_path": ""}
        if self.mode == "none":
            return paths

        stem = _case_stem(case_id)
        prediction = _label_mask(mask)
        want_polar = self.mode in {"polar", "both"}
        want_cartesian = self.mode in {"cartesian", "both"}

        cart_shape = None
        if want_cartesian:
            if self.cartesian_size == "original":
                if original_size is None:
                    raise ValueError(
                        "original_size=(H, W) is required when saving Cartesian masks "
                        "with prediction_cartesian_size='original'; use 'input' to save the input grid"
                    )
                cart_shape = _as_hw(original_size, "original_size", allow_square=False)
            else:
                cart_shape = _input_shape(self.cfg)

        prepared = []
        if want_polar:
            polar_mask = prediction if self.input_is_polar else _warp_mask(prediction, self.cfg, inverse=False)
            prepared.append(("pred_polar_path", polar_mask, self.polar_dir / f"{stem}.png"))
        if want_cartesian:
            cart_mask = _warp_mask(prediction, self.cfg, inverse=True) if self.input_is_polar else prediction
            cart_mask = _resize_mask(cart_mask, cart_shape)
            prepared.append(("pred_cartesian_path", cart_mask, self.cartesian_dir / f"{stem}.png"))

        for key, values, destination in prepared:
            _atomic_save_png(values, destination)
            paths[key] = str(destination)
        paths["pred_path"] = paths["pred_cartesian_path"] or paths["pred_polar_path"]
        return paths
