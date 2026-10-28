"""Coordinate-space and file-output contracts for saved prediction masks.

These tests use real OpenCV warps and PNG decoding, without importing a model,
Lightning, MONAI, or the Albumentations training pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.evaluation.prediction_masks import PredictionMaskWriter, normalize_save_mode


@pytest.mark.parametrize("mode", ["polar", "cartesian", "both"])
def test_explicit_directories_preserve_legacy_cli_layout(tmp_path, mode):
    cfg = _cfg(mode=mode)
    polar_dir = tmp_path / "model" / "masks_polar" / "dataset"
    cartesian_dir = tmp_path / "model" / "masks_cart" / "dataset"
    writer = PredictionMaskWriter(
        cfg, tmp_path / "unused_defaults", polar_dir=polar_dir, cartesian_dir=cartesian_dir,
    )
    paths = writer.save(_mask(cfg), "sample", original_size=(53, 79))
    assert polar_dir.exists() == (mode in {"polar", "both"})
    assert cartesian_dir.exists() == (mode in {"cartesian", "both"})
    assert not (tmp_path / "unused_defaults").exists()
    if mode in {"polar", "both"}:
        assert Path(paths["pred_polar_path"]) == polar_dir / "sample.png"
    if mode in {"cartesian", "both"}:
        assert Path(paths["pred_cartesian_path"]) == cartesian_dir / "sample.png"


def test_reference_csv_keeps_mask_paths_as_text(tmp_path):
    from src.evaluation.quantification_stats import read_reference_csv

    csv_path = tmp_path / "per_case.csv"
    csv_path.write_text(
        "case_id,pred_path,pred_polar_path,pred_cartesian_path\n"
        "001,/out/masks_polar/001.png,/out/masks_polar/001.png,\n"
        "002,0002,,0002\n",
        encoding="utf-8",
    )
    rows = read_reference_csv(csv_path)
    assert rows[0]["pred_cartesian_path"] == ""
    assert rows[1]["pred_polar_path"] == ""
    assert rows[1]["pred_cartesian_path"] == "0002"


def _cfg(*, polar=False, mode="both", size=32, geometry=None, cartesian_size="original"):
    geometry = {
        "center": None,
        "max_radius": None,
        "pt_type": "corner",
        "angle_bins": 48,
        "radius_bins": 40,
        "mode": "linear",
        **(geometry or {}),
    }
    return {
        "train": {"img_size": size},
        "polar": {"enabled": polar, "geometry": geometry},
        "eval": {
            "save_predictions": mode,
            "prediction_cartesian_size": cartesian_size,
        },
    }


def _mask(cfg):
    if cfg["polar"]["enabled"]:
        g = cfg["polar"]["geometry"]
        angle, radius = np.indices((g["angle_bins"], g["radius_bins"]))
        mask = np.zeros(angle.shape, dtype=np.int64)
        mask[radius < 8] = 1
        mask[(radius >= 8) & (radius < 14) & (angle < 18)] = 2
        return mask
    y, x = np.indices((cfg["train"]["img_size"],) * 2)
    distance = (x - 15) ** 2 + (y - 16) ** 2
    mask = np.zeros(y.shape, dtype=np.int64)
    mask[distance <= 36] = 1
    mask[(distance > 36) & (distance < 110) & (x > 15)] = 2
    return mask


def _geometry(cfg):
    size = cfg["train"]["img_size"]
    g = cfg["polar"]["geometry"]
    center = tuple(g["center"]) if g["center"] is not None else ((size - 1) / 2,) * 2
    radius = g["max_radius"]
    if radius is None:
        if g["pt_type"] == "min_edge":
            radius = min(center[0], center[1], size - 1 - center[0], size - 1 - center[1])
        else:
            corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
            radius = max(np.hypot(x - center[0], y - center[1]) for x, y in corners)
    return center, float(radius)


def _reference_warp(mask, cfg, *, inverse):
    """Use the public OpenCV contract, independently of writer helpers."""
    g = cfg["polar"]["geometry"]
    center, radius = _geometry(cfg)
    flags = cv2.INTER_NEAREST | cv2.WARP_FILL_OUTLIERS
    flags |= cv2.WARP_POLAR_LOG if g["mode"] == "log" else cv2.WARP_POLAR_LINEAR
    if inverse:
        flags |= cv2.WARP_INVERSE_MAP
        dsize = (cfg["train"]["img_size"],) * 2
    else:
        dsize = (g["radius_bins"], g["angle_bins"])
    return np.rint(cv2.warpPolar(
        mask.astype(np.float32), dsize, center, radius, flags
    )).astype(np.int64)


def _read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"PNG could not be read: {path}"
    assert image.ndim == 2
    return image


@pytest.mark.parametrize("value,expected", [
    (None, "none"), (False, "none"), (True, "both"),
    ("none", "none"), ("polar", "polar"), ("cartesian", "cartesian"),
    ("both", "both"), ("false", "none"), ("true", "both"), ("cart", "cartesian"),
])
def test_save_mode_normalization(value, expected):
    assert normalize_save_mode(value) == expected


@pytest.mark.parametrize("value", ["unknown", "polar_and_cart", [], {}])
def test_save_mode_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        normalize_save_mode(value)


@pytest.mark.parametrize("polar", [False, True], ids=["cartesian_input", "polar_input"])
@pytest.mark.parametrize("mode", ["none", "polar", "cartesian", "both"])
def test_four_modes_in_both_input_spaces(tmp_path, mode, polar):
    cfg = _cfg(polar=polar, mode=mode)
    mask = _mask(cfg)
    output = tmp_path / "predictions"
    result = PredictionMaskWriter(cfg, output).save(mask, "sample_007", original_size=(53, 79))
    want_polar = mode in {"polar", "both"}
    want_cart = mode in {"cartesian", "both"}

    assert bool(result["pred_polar_path"]) is want_polar
    assert bool(result["pred_cartesian_path"]) is want_cart
    assert (output / "masks_polar").exists() is want_polar
    assert (output / "masks_cartesian").exists() is want_cart
    assert result["pred_path"] == (result["pred_cartesian_path"] or result["pred_polar_path"])

    if want_polar:
        path = Path(result["pred_polar_path"])
        assert path.parent == output / "masks_polar"
        actual = _read(path)
        expected = mask if polar else _reference_warp(mask, cfg, inverse=False)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (48, 40)  # angle is height, radius is width
    if want_cart:
        path = Path(result["pred_cartesian_path"])
        assert path.parent == output / "masks_cartesian"
        actual = _read(path)
        base = _reference_warp(mask, cfg, inverse=True) if polar else mask
        expected = cv2.resize(base.astype(np.float32), (79, 53), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (53, 79)
        assert set(np.unique(actual)).issubset({0, 1, 2})
    if mode == "none":
        assert not output.exists()


def test_none_has_no_validation_conversion_or_file_side_effects(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled saving must not transform a mask")

    monkeypatch.setattr(cv2, "warpPolar", forbidden)
    monkeypatch.setattr(cv2, "resize", forbidden)
    output = tmp_path / "disabled"
    result = PredictionMaskWriter({"eval": {"save_predictions": None}}, output).save(
        None, "unused", original_size=None
    )
    assert result == {"pred_path": "", "pred_polar_path": "", "pred_cartesian_path": ""}
    assert not output.exists()


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("warp_mode", ["linear", "log"])
def test_custom_geometry_and_nonsquare_polar_grid(tmp_path, polar, warp_mode):
    cfg = _cfg(polar=polar, cartesian_size="input", geometry={
        "center": [12.25, 18.5], "max_radius": 27.75,
        "angle_bins": 36, "radius_bins": 52, "mode": warp_mode,
    })
    mask = _mask(cfg)
    result = PredictionMaskWriter(cfg, tmp_path).save(mask, "geometry")
    expected_polar = mask if polar else _reference_warp(mask, cfg, inverse=False)
    expected_cart = _reference_warp(mask, cfg, inverse=True) if polar else mask
    np.testing.assert_array_equal(_read(result["pred_polar_path"]), expected_polar)
    np.testing.assert_array_equal(_read(result["pred_cartesian_path"]), expected_cart)
    assert _read(result["pred_polar_path"]).shape == (36, 52)
    assert _read(result["pred_cartesian_path"]).shape == (32, 32)


def test_min_edge_geometry_uses_nearest_edge_from_custom_center(tmp_path):
    cfg = _cfg(mode="polar", geometry={"center": [10.0, 17.0], "pt_type": "min_edge"})
    mask = _mask(cfg)
    result = PredictionMaskWriter(cfg, tmp_path).save(mask, "min_edge")
    np.testing.assert_array_equal(_read(result["pred_polar_path"]), _reference_warp(mask, cfg, inverse=False))


@pytest.mark.parametrize("label,dtype", [(1, np.uint8), (255, np.uint8), (256, np.uint16), (65535, np.uint16)])
def test_png_preserves_semantic_labels_including_binary_one(tmp_path, label, dtype):
    cfg = _cfg(mode="cartesian", cartesian_size="input")
    mask = _mask(cfg)
    mask[mask != 0] = label
    result = PredictionMaskWriter(cfg, tmp_path).save(mask, "labels")
    decoded = _read(result["pred_path"])
    assert decoded.dtype == dtype
    np.testing.assert_array_equal(decoded, mask)
    assert set(np.unique(decoded)) == {0, label}


def test_constructor_mode_and_size_overrides_are_effective(tmp_path):
    cfg = _cfg(polar=True, mode="none", cartesian_size="original")
    result = PredictionMaskWriter(cfg, tmp_path, mode="cartesian", cartesian_size="input").save(
        _mask(cfg), "override"
    )
    assert result["pred_polar_path"] == ""
    assert _read(result["pred_path"]).shape == (32, 32)


def test_cartesian_default_config_needs_no_polar_geometry(tmp_path):
    cfg = {
        "train": {"img_size": 32},
        "polar": {"enabled": False},
        "eval": {"save_predictions": "both", "prediction_cartesian_size": "input"},
    }
    reference_cfg = _cfg(geometry={"angle_bins": 32, "radius_bins": 32})
    mask = _mask(reference_cfg)
    result = PredictionMaskWriter(cfg, tmp_path).save(mask, "cart_default")
    np.testing.assert_array_equal(_read(result["pred_cartesian_path"]), mask)
    np.testing.assert_array_equal(
        _read(result["pred_polar_path"]), _reference_warp(mask, reference_cfg, inverse=False)
    )


def test_original_size_is_required_only_for_cartesian_original_output(tmp_path):
    cfg = _cfg(polar=True, mode="polar")
    writer = PredictionMaskWriter(cfg, tmp_path / "polar_only")
    assert Path(writer.save(_mask(cfg), "case")["pred_polar_path"]).is_file()
    with pytest.raises(ValueError):
        PredictionMaskWriter(cfg, tmp_path / "cart", mode="cartesian").save(_mask(cfg), "case")


def test_repeated_case_is_atomically_replaced_without_temporary_files(tmp_path):
    cfg = _cfg(mode="cartesian", cartesian_size="input")
    writer = PredictionMaskWriter(cfg, tmp_path)
    initial = _mask(cfg)
    first = writer.save(initial, "same_case")
    replacement = np.full_like(initial, 2)
    second = writer.save(replacement, "same_case")
    assert first == second
    np.testing.assert_array_equal(_read(second["pred_path"]), replacement)
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == {Path(second["pred_path"])}


def test_failed_atomic_replace_preserves_existing_png(tmp_path, monkeypatch):
    cfg = _cfg(mode="cartesian", cartesian_size="input")
    writer = PredictionMaskWriter(cfg, tmp_path)
    initial = _mask(cfg)
    result = writer.save(initial, "same_case")
    path = Path(result["pred_path"])
    original_bytes = path.read_bytes()

    def refuse_replace(*args, **kwargs):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(os, "replace", refuse_replace)
    with pytest.raises(OSError, match="replacement failure"):
        writer.save(np.full_like(initial, 2), "same_case")
    assert path.read_bytes() == original_bytes
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == {path}


def test_concurrent_same_case_writes_leave_one_complete_png(tmp_path):
    cfg = _cfg(mode="cartesian", cartesian_size="input")
    writer = PredictionMaskWriter(cfg, tmp_path)
    masks = [np.full((32, 32), value, dtype=np.int64) for value in (1, 2, 3, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda mask: writer.save(mask, "same_case"), masks))
    assert all(row == results[0] for row in results)
    path = Path(results[0]["pred_path"])
    decoded = _read(path)
    assert any(np.array_equal(decoded, mask) for mask in masks)
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == {path}


def test_import_does_not_require_training_or_medical_metric_dependencies():
    root = Path(__file__).resolve().parents[1]
    script = """
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchmetrics', 'monai', 'albumentations', 'pytorch_lightning', 'lightning'}:
        raise AssertionError('Unexpected heavy dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from src.evaluation.prediction_masks import PredictionMaskWriter, normalize_save_mode
assert normalize_save_mode(False) == 'none'
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
