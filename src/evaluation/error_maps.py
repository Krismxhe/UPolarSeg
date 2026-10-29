"""Class-specific FP/FN maps from aligned integer prediction and GT labels.

Cartesian comparisons always use the original Cartesian reference on disk.
Polar batch GT is used only for polar comparisons. No probability threshold,
hard/soft quantification readout, or boundary postprocessing is involved.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.evaluation.fundus_overlays import load_fundus_rgb, resize_fundus_rgb
from src.evaluation.prediction_masks import (
    _as_hw,
    _atomic_save_png,
    _case_stem,
    _cfg_get,
    _input_shape,
    _label_mask,
    _positive_int,
    _resize_mask,
    _resolve_geometry,
    _warp_mask,
    normalize_save_mode,
)


ERROR_MAP_PATH_FIELDS = (
    "error_od_polar_path", "error_ppa_polar_path",
    "error_od_cartesian_path", "error_ppa_cartesian_path",
    "error_od_fundus_path", "error_ppa_fundus_path",
    "error_metadata_path",
)
_COLORS = {"fp": (255, 0, 0), "fn": (0, 0, 255), "other": (0, 0, 0)}


def normalize_error_mode(value) -> str:
    """Normalize the four pure-error output spaces, including legacy booleans."""
    try:
        return normalize_save_mode(value)
    except ValueError as exc:
        raise ValueError("eval.save_error_maps must be none/polar/cartesian/both") from exc


def normalize_error_classes(value) -> str:
    if isinstance(value, str) and value.strip().lower() in {"od", "ppa", "both"}:
        return value.strip().lower()
    raise ValueError("eval.error_map_classes must be od/ppa/both")


def _strict_bool(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} must be a boolean or the string true/false")


def parse_error_fundus_flag(value) -> bool:
    """Reject truthy strings/numbers rather than accidentally enabling export."""
    return _strict_bool(value, "eval.save_error_fundus")


def _reference_complete(value):
    if value is None or (isinstance(value, str) and value == ""):
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f"}:
            return False
        raise ValueError(f"Invalid reference_complete flag: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Invalid reference_complete flag: {value!r}")


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            if hasattr(value, "float"):
                return value.float().numpy()
            raise
    return np.asarray(value)


def _labels(value, num_classes, name):
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        labels = _label_mask(_numpy(value))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    if int(labels.max()) >= num_classes:
        raise ValueError(
            f"{name} labels must lie in 0..{num_classes - 1}; "
            "class-index labels are not inferred from 0/127/255 encodings"
        )
    return labels


def _native_groundtruth(mask_path, num_classes):
    if not isinstance(mask_path, (str, os.PathLike)) or not str(mask_path).strip():
        raise ValueError("mask_path is required for Cartesian/fundus error maps")
    try:
        with Image.open(Path(mask_path)) as image:
            mask = np.array(image, copy=True)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Could not read original Cartesian GT mask_path {str(mask_path)!r}: {exc}") from exc
    if mask.ndim == 3:
        if mask.shape[2] != 3 or not (
            np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 0], mask[..., 2])
        ):
            raise ValueError("Original Cartesian GT must be single-channel or RGB with three identical channels")
        mask = mask[..., 0]
    return _labels(mask, num_classes, "original Cartesian GT")


def _compare(prediction, reference, class_id):
    predicted = prediction == class_id
    target = reference == class_id
    fp = predicted & ~target
    fn = ~predicted & target
    counts = {
        "tp": int(np.count_nonzero(predicted & target)),
        "fp": int(np.count_nonzero(fp)),
        "fn": int(np.count_nonzero(fn)),
        "tn": int(np.count_nonzero(~predicted & ~target)),
    }
    return counts, fp, fn


def _pure_errors(fp, fn):
    canvas = np.zeros((*fp.shape, 3), dtype=np.uint8)
    canvas[fp] = _COLORS["fp"]
    canvas[fn] = _COLORS["fn"]
    return canvas


def _fundus_errors(rgb, fp, fn, alpha):
    canvas = rgb.copy()
    for selected, key in ((fp, "fp"), (fn, "fn")):
        color = np.asarray(_COLORS[key], dtype=np.float64)
        blended = (1.0 - alpha) * rgb[selected].astype(np.float64) + alpha * color
        canvas[selected] = np.rint(blended).astype(np.uint8)
    return canvas


def _fundus_legend(canvas, fp, fn, name, counts, source):
    """Place a compact legend in the corner containing the fewest errors."""
    h, w = canvas.shape[:2]
    rows = [((235, 239, 245), f"{name.upper()} errors | {source}"),
            ((255, 96, 96), f"Red: FP {counts['fp']} px"),
            ((110, 160, 255), f"Blue: FN {counts['fn']} px")]
    font_size = min(14, max(10, min(h, w) // 40))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    line_height = font_size + 4
    box_width = max(font.getbbox(text)[2] for _, text in rows) + 16
    box_height = len(rows) * line_height + 10
    if box_width + 8 > w or box_height + 8 > h or box_height > h * 0.45:
        return canvas
    candidates = ((4, 4), (w - box_width - 4, 4),
                  (4, h - box_height - 4), (w - box_width - 4, h - box_height - 4))
    errors = fp | fn
    x, y = min(candidates, key=lambda point: int(np.count_nonzero(
        errors[point[1]:point[1] + box_height, point[0]:point[0] + box_width]
    )))
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((x, y, x + box_width, y + box_height), radius=4, fill=(12, 16, 22))
    for index, (color, text) in enumerate(rows):
        draw.text((x + 8, y + 4 + index * line_height), text, font=font, fill=color)
    return np.array(image)


def _atomic_json(encoded, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class ErrorMapWriter:
    """Export per-class errors on the exact grid on which each comparison runs."""

    def __init__(self, cfg, output_dir, *, polar_dir=None, cartesian_dir=None,
                 fundus_dir=None, metadata_dir=None):
        self.cfg = cfg
        evaluation = _cfg_get(cfg, "eval", {}) or {}
        self.mode = normalize_error_mode(_cfg_get(evaluation, "save_error_maps", "none"))
        self.save_fundus = parse_error_fundus_flag(_cfg_get(evaluation, "save_error_fundus", False))
        self.active = self.mode != "none" or self.save_fundus
        self.classes = normalize_error_classes(_cfg_get(evaluation, "error_map_classes", "both"))
        self.source = str(_cfg_get(evaluation, "error_map_source", "seg_logits")).strip().lower()
        if self.source not in {"seg_logits", "quantification"}:
            raise ValueError("eval.error_map_source must be seg_logits or quantification")
        self.cartesian_size = str(_cfg_get(evaluation, "prediction_cartesian_size", "original")).strip().lower()
        if self.cartesian_size not in {"original", "input"}:
            raise ValueError("eval.prediction_cartesian_size must be original or input")
        try:
            self.alpha = float(_cfg_get(evaluation, "error_map_alpha", 0.6))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("eval.error_map_alpha must be a finite number in [0, 1]") from exc
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("eval.error_map_alpha must be a finite number in [0, 1]")
        self.legend = _strict_bool(_cfg_get(evaluation, "error_map_legend", True), "eval.error_map_legend")
        output_dir = Path(output_dir)
        self.polar_dir = Path(polar_dir) if polar_dir is not None else output_dir / "errors_polar"
        self.cartesian_dir = Path(cartesian_dir) if cartesian_dir is not None else output_dir / "errors_cartesian"
        self.fundus_dir = Path(fundus_dir) if fundus_dir is not None else output_dir / "errors_fundus"
        self.metadata_dir = Path(metadata_dir) if metadata_dir is not None else output_dir / "error_metadata"
        self.class_ids = {}
        self.num_classes = None
        # Disabled exports do not impose segmentation/geometry constraints on
        # unrelated training configurations or touch their input files.
        if self.active:
            dataset = _cfg_get(cfg, "dataset", {}) or {}
            if _cfg_get(dataset, "mask_mode", None) != "index":
                raise ValueError("Error maps require dataset.mask_mode=index")
            self.num_classes = _positive_int(_cfg_get(dataset, "num_classes", None), "dataset.num_classes")
            quantification = _cfg_get(cfg, "quantification", {}) or {}
            self.class_ids = {
                "od": _positive_int(_cfg_get(quantification, "od_class", 1), "quantification.od_class"),
                "ppa": _positive_int(_cfg_get(quantification, "ppa_class", 2), "quantification.ppa_class"),
            }
            if self.class_ids["od"] == self.class_ids["ppa"]:
                raise ValueError("OD and PPA class IDs must be distinct")
            if max(self.class_ids.values()) >= self.num_classes:
                raise ValueError("OD/PPA class IDs must be positive and smaller than dataset.num_classes")
            odppa = _cfg_get(evaluation, "odppa", {}) or {}
            if bool(_cfg_get(odppa, "enabled", False)):
                metric_class_ids = {
                    "od": _positive_int(_cfg_get(odppa, "od_class_id", 1), "eval.odppa.od_class_id"),
                    "ppa": _positive_int(_cfg_get(odppa, "ppa_class_id", 2), "eval.odppa.ppa_class_id"),
                }
                if metric_class_ids != self.class_ids:
                    raise ValueError(
                        f"Enabled eval.odppa class mapping {metric_class_ids} differs from "
                        f"quantification class mapping {self.class_ids} used by error maps; "
                        "set eval.odppa.od_class_id/ppa_class_id and "
                        "quantification.od_class/ppa_class consistently"
                    )

    @property
    def needs_original_size(self):
        return self.cartesian_size == "original" and (
            self.mode in {"cartesian", "both"} or self.save_fundus
        )

    def save(self, mask, case_id, *, groundtruth_mask=None, mask_path=None, image_path=None,
             original_size=None, reference_complete=True, prediction_source="seg_logits"):
        paths = {key: "" for key in ERROR_MAP_PATH_FIELDS}
        if not self.active:
            return paths
        if prediction_source not in {"seg_logits", "quant_logits"}:
            raise ValueError("prediction_source must be seg_logits or quant_logits")
        stem = _case_stem(case_id)
        complete = _reference_complete(reference_complete)
        prediction = _labels(mask, self.num_classes, "prediction mask")
        input_shape = _input_shape(self.cfg)
        input_is_polar = bool(_cfg_get(_cfg_get(self.cfg, "polar", {}), "enabled", False))
        want_polar = self.mode in {"polar", "both"}
        want_cartesian = self.mode in {"cartesian", "both"}
        need_cartesian = want_cartesian or self.save_fundus
        geometry = None
        if input_is_polar or want_polar:
            cart_shape, polar_shape, center, radius, polar_mode = _resolve_geometry(self.cfg)
            geometry = {"center_xy": list(center), "max_radius": radius, "mode": polar_mode,
                        "cartesian_shape": list(cart_shape), "polar_shape": list(polar_shape)}
        expected_shape = polar_shape if input_is_polar else input_shape
        if prediction.shape != expected_shape:
            raise ValueError(f"Prediction model grid {prediction.shape} differs from configured grid {expected_shape}")
        batch_gt = None
        if groundtruth_mask is not None:
            batch_gt = _labels(groundtruth_mask, self.num_classes, "batch groundtruth_mask")
            if batch_gt.shape != prediction.shape:
                raise ValueError("Prediction and batch groundtruth_mask must have the same model-grid shape")
        if want_polar and batch_gt is None:
            raise ValueError("Polar error maps require batch groundtruth_mask on the model grid")
        declared_shape = None if original_size is None else _as_hw(original_size, "original_size", allow_square=False)
        native_gt = None
        original_shape = None
        cartesian_shape = None
        rgb = None
        if need_cartesian:
            native_gt = _native_groundtruth(mask_path, self.num_classes)
            original_shape = tuple(native_gt.shape)
            if declared_shape is not None and declared_shape != original_shape:
                raise ValueError(
                    f"original_size={declared_shape} does not match original Cartesian GT size {original_shape}"
                )
            cartesian_shape = original_shape if self.cartesian_size == "original" else input_shape
            if self.save_fundus:
                rgb = load_fundus_rgb(image_path)
                if tuple(rgb.shape[:2]) != original_shape:
                    raise ValueError(
                        f"Fundus image size {tuple(rgb.shape[:2])} does not match original Cartesian GT size {original_shape}"
                    )
                rgb = resize_fundus_rgb(rgb, cartesian_shape)

        comparisons = {}
        if want_polar:
            if input_is_polar:
                polar_prediction, polar_gt = prediction, batch_gt
                gt_source = "batch_polar"
                transform = "native_polar"
            else:
                polar_prediction = _warp_mask(prediction, self.cfg, inverse=False)
                polar_gt = _warp_mask(batch_gt, self.cfg, inverse=False)
                gt_source = "batch_cartesian_forward_warp"
                transform = "cartesian_forward_warp"
            comparisons["polar"] = (polar_prediction, polar_gt, gt_source, transform)
        if need_cartesian:
            cartesian_prediction = _warp_mask(prediction, self.cfg, inverse=True) if input_is_polar else prediction
            cartesian_prediction = _resize_mask(cartesian_prediction, cartesian_shape)
            # Resize the disk reference directly, once. Never inverse-warp the
            # polar batch GT or resize a precomputed categorical error map.
            cartesian_gt = _resize_mask(native_gt, cartesian_shape)
            transform = "inverse_polar_then_nearest_resize" if input_is_polar else "nearest_resize"
            comparisons["cartesian"] = (cartesian_prediction, cartesian_gt, "native_cartesian_mask", transform)

        selected = ("od", "ppa") if self.classes == "both" else (self.classes,)
        evaluated = {}
        for space, (predicted, target, _, _) in comparisons.items():
            evaluated[space] = {
                name: _compare(predicted, target, self.class_ids[name])
                for name in selected if name != "ppa" or complete
            }
        metadata = {
            "schema_version": "error-map-v1", "case_id": stem,
            "selected_source": self.source, "actual_source": prediction_source,
            "mode": self.mode, "save_fundus": self.save_fundus, "classes": self.classes,
            "mask_path": str(mask_path) if mask_path is not None else None,
            "image_path": str(image_path) if image_path is not None else None,
            "model_space": "polar" if input_is_polar else "cartesian",
            "model_shape": list(prediction.shape), "cartesian_input_shape": list(input_shape),
            "original_shape": list(original_shape) if original_shape is not None else None,
            "declared_original_shape": list(declared_shape) if declared_shape is not None else None,
            "cartesian_size": self.cartesian_size, "geometry": geometry,
            "class_ids": dict(self.class_ids), "colors_rgb": {key: list(value) for key, value in _COLORS.items()},
            "reference_complete": complete, "count_unit": "comparison_grid_pixels",
            "spaces": {}, "paths": paths,
        }
        prepared = []
        for requested, space, directory in (
            (want_polar, "polar", self.polar_dir),
            (want_cartesian, "cartesian", self.cartesian_dir),
            (self.save_fundus, "fundus", self.fundus_dir),
        ):
            if not requested:
                continue
            comparison_space = "cartesian" if space == "fundus" else space
            predicted, _, gt_source, transform = comparisons[comparison_space]
            space_record = {"shape": list(predicted.shape), "groundtruth_source": gt_source,
                            "prediction_transform": transform, "count_unit": "comparison_grid_pixels", "classes": {}}
            if space == "fundus":
                space_record.update({"alpha": self.alpha, "legend": self.legend})
            for name in selected:
                if name == "ppa" and not complete:
                    space_record["classes"][name] = {
                        "class_id": self.class_ids[name], "status": "not_evaluable",
                        "reason": "incomplete_reference", "counts": None, "path": "",
                    }
                    continue
                counts, fp, fn = evaluated[comparison_space][name]
                if space == "fundus":
                    canvas = _fundus_errors(rgb, fp, fn, self.alpha)
                    if self.legend:
                        canvas = _fundus_legend(canvas, fp, fn, name, counts, prediction_source)
                else:
                    canvas = _pure_errors(fp, fn)
                destination = directory / name / f"{stem}.png"
                key = f"error_{name}_{space}_path"
                paths[key] = str(destination)
                prepared.append((canvas, destination))
                space_record["classes"][name] = {
                    "class_id": self.class_ids[name], "status": "evaluable", "reason": None,
                    "counts": dict(counts), "path": str(destination),
                }
            metadata["spaces"][space] = space_record
        destinations = [destination.resolve() for _, destination in prepared]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Selected error map output paths must be distinct")
        metadata_path = self.metadata_dir / f"{stem}.json"
        paths["error_metadata_path"] = str(metadata_path)
        # Finish validation and strict serialization before any directory/file
        # creation, including cases where every selected PPA result is skipped.
        encoded = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
        for canvas, destination in prepared:
            _atomic_save_png(canvas, destination)
        _atomic_json(encoded, metadata_path)
        return paths
