"""Error-map pixel contracts with independent label/geometry comparisons.

The reference code invokes OpenCV with explicit geometry; it never uses the
writer's resize, warp, comparison, or rendering helpers as its oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from src.evaluation.error_maps import (
    ErrorMapWriter,
    normalize_error_classes,
    normalize_error_mode,
    parse_error_fundus_flag,
)


PATH_FIELDS = {
    f"error_{label}_{space}_path"
    for label in ("od", "ppa") for space in ("polar", "cartesian", "fundus")
} | {"error_metadata_path"}
GT = np.array([[0, 1, 1, 2], [0, 1, 2, 2], [0, 0, 0, 0], [2, 2, 1, 0]], dtype=np.uint8)
PRED = np.array([[0, 1, 2, 2], [1, 0, 2, 0], [0, 2, 0, 0], [1, 2, 1, 0]], dtype=np.uint8)


def _cfg(*, mode="cartesian", fundus=False, polar=False, size=4, output_size="original",
         classes="both", alpha=0.6, polar_mode="linear"):
    return {
        "train": {"img_size": size},
        "dataset": {"num_classes": 3, "mask_mode": "index"},
        "quantification": {"od_class": 1, "ppa_class": 2},
        "polar": {"enabled": polar, "geometry": {
            "center": [(size - 1) * 0.35, (size - 1) * 0.6],
            "max_radius": size * 0.75, "radius_bins": size + 3,
            "angle_bins": size + 5, "mode": polar_mode, "pt_type": "corner",
        }},
        "eval": {
            "save_error_maps": mode, "save_error_fundus": fundus,
            "error_map_classes": classes, "error_map_source": "seg_logits",
            "error_map_alpha": alpha, "error_map_legend": False,
            "prediction_cartesian_size": output_size,
            "odppa": {"od_class_id": 1, "ppa_class_id": 2},
        },
    }


def _pattern(shape):
    y, x = np.indices(shape)
    return np.where((x + 2 * y) % 5 == 0, 1, np.where((2 * x + y) % 7 == 0, 2, 0)).astype(np.uint8)


def _forward(mask, cfg):
    size = cfg["train"]["img_size"]
    g = cfg["polar"]["geometry"]
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    flag = cv2.WARP_POLAR_LOG if g["mode"] == "log" else cv2.WARP_POLAR_LINEAR
    return cv2.warpPolar(
        resized, (g["radius_bins"], g["angle_bins"]), tuple(g["center"]), g["max_radius"],
        flag | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST,
    )


def _inverse(mask, cfg):
    size = cfg["train"]["img_size"]
    g = cfg["polar"]["geometry"]
    flag = cv2.WARP_POLAR_LOG if g["mode"] == "log" else cv2.WARP_POLAR_LINEAR
    return cv2.warpPolar(
        mask, (size, size), tuple(g["center"]), g["max_radius"],
        flag | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST,
    )


def _nearest(mask, shape):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _counts(pred, gt, class_id):
    p, g = pred == class_id, gt == class_id
    return {"tp": int((p & g).sum()), "fp": int((p & ~g).sum()),
            "fn": int((~p & g).sum()), "tn": int((~p & ~g).sum())}


def _colors(pred, gt, class_id):
    output = np.zeros((*pred.shape, 3), dtype=np.uint8)
    output[(pred == class_id) & (gt != class_id)] = (255, 0, 0)
    output[(pred != class_id) & (gt == class_id)] = (0, 0, 255)
    return output


def _rgb(tmp_path, shape):
    y, x = np.indices(shape)
    array = np.stack((30 + x % 40, 60 + y % 50, 90 + (x + y) % 60), axis=-1).astype(np.uint8)
    path = tmp_path / "native_rgb.png"
    Image.fromarray(array).save(path)
    return path, array


def _native(tmp_path, mask):
    path = tmp_path / "native_gt.png"
    Image.fromarray(mask).save(path)
    return path


def _metadata(path):
    def reject(value):
        pytest.fail(f"Nonstandard numeric constant in JSON: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _assert_space(metadata, space, pred, gt, *, labels=("od", "ppa")):
    record = metadata["spaces"][space]
    assert record["shape"] == list(pred.shape)
    assert record["count_unit"] == "comparison_grid_pixels"
    assert set(record["classes"]) == set(labels)
    for label in labels:
        item = record["classes"][label]
        class_id = metadata["class_ids"][label]
        assert item["class_id"] == class_id
        assert item["status"] == "evaluable"
        assert item["reason"] is None
        assert item["counts"] == _counts(pred, gt, class_id)
        assert sum(item["counts"].values()) == pred.size
        assert all(type(value) is int for value in item["counts"].values())
        if space != "fundus":
            with Image.open(item["path"]) as exported:
                assert exported.mode == "RGB"
                np.testing.assert_array_equal(np.asarray(exported), _colors(pred, gt, class_id))


def test_exact_confusion_pixels_are_rgb_and_keep_cross_class_errors_separate(tmp_path):
    cfg = _cfg()
    cfg["eval"]["error_map_legend"] = True  # Never draws a legend on pure RGB errors.
    mask_path = _native(tmp_path, GT)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        PRED, "case", groundtruth_mask=GT, mask_path=mask_path,
    )
    assert set(paths) == PATH_FIELDS
    meta = _metadata(paths["error_metadata_path"])
    assert meta["schema_version"] == "error-map-v1"
    assert meta["count_unit"] == "comparison_grid_pixels"
    assert meta["colors_rgb"] == {"fp": [255, 0, 0], "fn": [0, 0, 255], "other": [0, 0, 0]}
    assert meta["paths"] == paths
    _assert_space(meta, "cartesian", PRED, GT)
    assert meta["spaces"]["cartesian"]["classes"]["od"]["counts"] == {"tp": 2, "fp": 2, "fn": 2, "tn": 10}
    assert meta["spaces"]["cartesian"]["classes"]["ppa"]["counts"] == {"tp": 3, "fp": 2, "fn": 2, "tn": 9}
    od = np.asarray(Image.open(paths["error_od_cartesian_path"]))
    ppa = np.asarray(Image.open(paths["error_ppa_cartesian_path"]))
    np.testing.assert_array_equal(od[0, 2], [0, 0, 255])
    np.testing.assert_array_equal(ppa[0, 2], [255, 0, 0])


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("mode", ["none", "polar", "cartesian", "both"])
def test_four_modes_use_the_requested_grid_and_only_create_requested_files(tmp_path, polar, mode):
    cfg = _cfg(mode=mode, polar=polar, size=8)
    shape = (13, 11) if polar else (8, 8)
    gt = _pattern(shape)
    pred = np.roll(gt, 1, axis=1)
    native = _pattern((12, 18))
    mask_path = _native(tmp_path, native)
    root = tmp_path / "out"
    paths = ErrorMapWriter(cfg, root).save(pred, "case", groundtruth_mask=gt, mask_path=mask_path)
    if mode == "none":
        assert not any(paths.values())
        assert not root.exists()
        return
    meta = _metadata(paths["error_metadata_path"])
    assert meta["model_space"] == ("polar" if polar else "cartesian")
    assert meta["model_shape"] == list(shape)
    expected_spaces = {space for space in ("polar", "cartesian") if mode in {space, "both"}}
    assert set(meta["spaces"]) == expected_spaces
    for space in ("polar", "cartesian"):
        requested = space in expected_spaces
        for label in ("od", "ppa"):
            assert bool(paths[f"error_{label}_{space}_path"]) is requested
        if not requested:
            assert not (root / f"errors_{space}").exists()
            continue
        if space == "polar":
            expected_pred, expected_gt = (pred, gt) if polar else (_forward(pred, cfg), _forward(gt, cfg))
            expected_source = "batch_polar" if polar else "batch_cartesian_forward_warp"
        else:
            expected_pred = _nearest(_inverse(pred, cfg) if polar else pred, native.shape)
            expected_gt, expected_source = native, "native_cartesian_mask"
        assert meta["spaces"][space]["groundtruth_source"] == expected_source
        _assert_space(meta, space, expected_pred, expected_gt)
    assert not paths["error_od_fundus_path"] and not paths["error_ppa_fundus_path"]


@pytest.mark.parametrize("polar_mode", ["linear", "log"])
def test_native_comparison_keeps_real_restoration_error_instead_of_inverse_warping_batch_gt(tmp_path, polar_mode):
    cfg = _cfg(mode="both", polar=True, size=16, polar_mode=polar_mode)
    cfg["polar"]["geometry"].update(center=[5.25, 8.5], angle_bins=9, radius_bins=7, max_radius=8.0)
    native = _pattern((23, 31))
    native[-1, -1] = 2  # Outside this off-centre polar disk: necessarily lost on restoration.
    polar_gt = _forward(native, cfg)
    prediction = polar_gt.copy()  # Perfect on the model grid, not necessarily native pixels.
    mask_path = _native(tmp_path, native)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        prediction, "case", groundtruth_mask=polar_gt, mask_path=mask_path,
    )
    meta = _metadata(paths["error_metadata_path"])
    _assert_space(meta, "polar", prediction, polar_gt)
    restored = _nearest(_inverse(prediction, cfg), native.shape)
    _assert_space(meta, "cartesian", restored, native)
    assert meta["geometry"]["center_xy"] == [5.25, 8.5]
    assert meta["geometry"]["mode"] == polar_mode
    assert meta["spaces"]["polar"]["classes"]["ppa"]["counts"]["fn"] == 0
    assert meta["spaces"]["cartesian"]["classes"]["ppa"]["counts"]["fn"] > 0
    assert np.any(restored != native)


@pytest.mark.parametrize("output_size", ["input", "original"])
def test_resize_labels_before_comparison_and_count_in_the_actual_output_grid(tmp_path, output_size):
    cfg = _cfg(output_size=output_size)
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[2, 1] = 1
    native = np.zeros((8, 12), dtype=np.uint8)
    native[1, 1] = 1  # Not sampled by 8x12 -> 4x4 nearest resize.
    native[6, 9] = 2
    mask_path = _native(tmp_path, native)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        pred, "case", groundtruth_mask=np.zeros_like(pred), mask_path=mask_path,
        original_size=(8, 12),
    )
    meta = _metadata(paths["error_metadata_path"])
    shape = (4, 4) if output_size == "input" else (8, 12)
    expected_pred = _nearest(pred, shape)
    expected_gt = _nearest(native, shape)
    _assert_space(meta, "cartesian", expected_pred, expected_gt)
    counts = meta["spaces"]["cartesian"]["classes"]["od"]["counts"]
    assert counts["fp"] == (1 if output_size == "input" else 6)
    assert counts["fn"] == (0 if output_size == "input" else 1)


@pytest.mark.parametrize("classes", ["od", "ppa", "both"])
@pytest.mark.parametrize("complete", [True, False])
def test_complete_negative_ppa_has_false_positives_but_incomplete_reference_is_not_evaluable(
    tmp_path, classes, complete,
):
    cfg = _cfg(mode="both", polar=True, size=4, classes=classes)
    cfg["polar"]["geometry"].update(angle_bins=4, radius_bins=4)
    pred, gt = PRED.copy(), (GT == 1).astype(np.uint8)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        pred, "case", groundtruth_mask=gt, mask_path=_native(tmp_path, gt),
        reference_complete="true" if complete else "false",
    )
    meta = _metadata(paths["error_metadata_path"])
    assert meta["reference_complete"] is complete
    for space in ("polar", "cartesian"):
        records = meta["spaces"][space]["classes"]
        selected = {classes} if classes != "both" else {"od", "ppa"}
        assert set(records) == selected
        for label in ("od", "ppa"):
            path = paths[f"error_{label}_{space}_path"]
            if label not in selected:
                assert path == ""
            elif label == "ppa" and not complete:
                assert path == records[label]["path"] == ""
                assert records[label]["status"] == "not_evaluable"
                assert records[label]["reason"] == "incomplete_reference"
                assert records[label]["counts"] is None
            else:
                assert Path(path).is_file()
                assert records[label]["status"] == "evaluable"
    if complete and classes in {"ppa", "both"}:
        counts = meta["spaces"]["polar"]["classes"]["ppa"]["counts"]
        assert counts["fp"] == int((pred == 2).sum()) > 0
        assert counts["fn"] == counts["tp"] == 0


@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_fundus_only_blends_only_fp_fn_pixels_and_preserves_inputs(tmp_path, alpha):
    cfg = _cfg(mode="none", fundus=True, alpha=alpha)
    image_path, rgb = _rgb(tmp_path, GT.shape)
    mask_path = _native(tmp_path, GT)
    image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
    pred, batch_gt = PRED.copy(), GT.copy()
    root = tmp_path / "out"
    paths = ErrorMapWriter(cfg, root).save(
        pred, "case", groundtruth_mask=batch_gt, mask_path=mask_path, image_path=image_path,
    )
    meta = _metadata(paths["error_metadata_path"])
    assert set(meta["spaces"]) == {"fundus"}
    assert meta["spaces"]["fundus"]["alpha"] == alpha
    _assert_space(meta, "fundus", pred, GT)
    for label, class_id in (("od", 1), ("ppa", 2)):
        colors = _colors(pred, GT, class_id)
        incorrect = np.any(colors != 0, axis=-1)
        expected = rgb.copy()
        expected[incorrect] = np.rint((1-alpha) * rgb[incorrect] + alpha * colors[incorrect]).astype(np.uint8)
        exported = np.asarray(Image.open(paths[f"error_{label}_fundus_path"]))
        np.testing.assert_allclose(exported, expected, atol=1, rtol=0)
        np.testing.assert_array_equal(exported[~incorrect], rgb[~incorrect])
        if alpha == 0:
            np.testing.assert_array_equal(exported, rgb)
    assert all(paths[f"error_{label}_{space}_path"] == "" for label in ("od", "ppa") for space in ("polar", "cartesian"))
    np.testing.assert_array_equal(pred, PRED)
    np.testing.assert_array_equal(batch_gt, GT)
    assert image_path.read_bytes() == image_bytes
    assert mask_path.read_bytes() == mask_bytes


def test_cartesian_and_fundus_counts_share_native_gt_and_same_grid(tmp_path):
    cfg = _cfg(fundus=True)
    image_path, _ = _rgb(tmp_path, GT.shape)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        PRED, "case", groundtruth_mask=np.zeros_like(GT), mask_path=_native(tmp_path, GT), image_path=image_path,
    )
    meta = _metadata(paths["error_metadata_path"])
    _assert_space(meta, "cartesian", PRED, GT)
    _assert_space(meta, "fundus", PRED, GT)
    for label in ("od", "ppa"):
        assert meta["spaces"]["cartesian"]["classes"][label]["counts"] == meta["spaces"]["fundus"]["classes"][label]["counts"]


def test_custom_class_ids_and_destinations_do_not_assume_three_class_order(tmp_path):
    cfg = _cfg(mode="both")
    cfg["dataset"]["num_classes"] = 5
    cfg["quantification"].update(od_class=4, ppa_class=1)
    pred = np.where(PRED == 1, 4, np.where(PRED == 2, 1, 0)).astype(np.uint8)
    gt = np.where(GT == 1, 4, np.where(GT == 2, 1, 0)).astype(np.uint8)
    writer = ErrorMapWriter(cfg, tmp_path / "out", polar_dir=tmp_path / "polar_custom",
                            cartesian_dir=tmp_path / "cart_custom", metadata_dir=tmp_path / "json_custom")
    paths = writer.save(pred, "case", groundtruth_mask=gt, mask_path=_native(tmp_path, gt))
    meta = _metadata(paths["error_metadata_path"])
    assert meta["class_ids"] == {"od": 4, "ppa": 1}
    _assert_space(meta, "cartesian", pred, gt)
    _assert_space(meta, "polar", _forward(pred, cfg), _forward(gt, cfg))
    assert Path(paths["error_od_polar_path"]) == tmp_path / "polar_custom" / "od" / "case.png"
    assert Path(paths["error_ppa_cartesian_path"]) == tmp_path / "cart_custom" / "ppa" / "case.png"
    assert Path(paths["error_metadata_path"]) == tmp_path / "json_custom" / "case.json"


def test_disabled_is_a_true_noop_without_geometry_dataset_or_file_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled error maps must not read any input file")
    monkeypatch.setattr(Image, "open", forbidden)
    root = tmp_path / "disabled"
    writer = ErrorMapWriter({}, root)
    paths = writer.save(None, "../unused", groundtruth_mask=None, mask_path=object(), image_path=object(),
                        original_size="invalid", prediction_source=None)
    assert paths == dict.fromkeys(PATH_FIELDS, "")
    assert writer.active is False
    assert not root.exists()


def test_polar_only_uses_batch_gt_without_reading_native_or_rgb_files(tmp_path, monkeypatch):
    cfg = _cfg(mode="polar", polar=True)
    cfg["polar"]["geometry"].update(angle_bins=4, radius_bins=4)
    def forbidden(*args, **kwargs):
        raise AssertionError("Direct polar comparison requires no disk GT or RGB")
    monkeypatch.setattr(Image, "open", forbidden)
    paths = ErrorMapWriter(cfg, tmp_path / "out").save(
        PRED, "case", groundtruth_mask=GT, mask_path="missing.png", image_path="missing.jpg",
    )
    assert Path(paths["error_od_polar_path"]).is_file()


@pytest.mark.parametrize("native_kind", ["absent", "missing", "rgb", "unknown_label"])
def test_native_gt_is_required_and_malformed_gt_never_becomes_zero_error(tmp_path, native_kind):
    mask_path = None
    if native_kind == "missing":
        mask_path = tmp_path / "missing.png"
    elif native_kind == "rgb":
        mask_path = _native(tmp_path, np.dstack((GT, np.zeros_like(GT), GT)))
    elif native_kind == "unknown_label":
        mask_path = _native(tmp_path, np.full_like(GT, 7))
    root = tmp_path / "out"
    with pytest.raises((ValueError, OSError)):
        ErrorMapWriter(_cfg(), root).save(PRED, "case", groundtruth_mask=GT, mask_path=mask_path)
    assert not list(root.rglob("*.png"))


@pytest.mark.parametrize("kind", ["prediction_probability", "batch_shape", "batch_probability", "missing_polar_gt"])
def test_polar_comparison_rejects_invalid_model_labels_and_target_grid(tmp_path, kind):
    cfg = _cfg(mode="polar", polar=True)
    cfg["polar"]["geometry"].update(angle_bins=4, radius_bins=4)
    prediction, target = PRED, GT
    if kind == "prediction_probability":
        prediction = PRED.astype(float) + 0.25
    elif kind == "batch_shape":
        target = np.zeros((3, 4), dtype=np.uint8)
    elif kind == "batch_probability":
        target = GT.astype(float) + 0.25
    else:
        target = None
    with pytest.raises(ValueError):
        ErrorMapWriter(cfg, tmp_path / "out").save(prediction, "case", groundtruth_mask=target)


@pytest.mark.parametrize("output_size", ["input", "original"])
def test_native_size_and_rgb_dimensions_must_match_declared_metadata(tmp_path, output_size):
    cfg = _cfg(fundus=True, output_size=output_size)
    mask_path = _native(tmp_path, GT)
    image_path, _ = _rgb(tmp_path, (4, 5))
    with pytest.raises(ValueError, match="size|shape|dimension"):
        ErrorMapWriter(cfg, tmp_path / "wrong_rgb").save(
            PRED, "case", groundtruth_mask=GT, mask_path=mask_path, image_path=image_path,
        )
    cfg["eval"]["save_error_fundus"] = False
    with pytest.raises(ValueError, match="size|shape|dimension"):
        ErrorMapWriter(cfg, tmp_path / "wrong_meta").save(
            PRED, "case", groundtruth_mask=GT, mask_path=mask_path, original_size=(8, 8),
        )


@pytest.mark.parametrize("setting,value", [
    ("num_classes", 2), ("mask_mode", "binary"),
    ("od_class", 0), ("od_class", 2), ("ppa_class", 3), ("ppa_class", 1.5),
])
def test_active_export_rejects_unsupported_dataset_or_class_ids(tmp_path, setting, value):
    cfg = _cfg()
    node = cfg["dataset"] if setting in {"num_classes", "mask_mode"} else cfg["quantification"]
    node[setting] = value
    with pytest.raises(ValueError):
        ErrorMapWriter(cfg, tmp_path)


@pytest.mark.parametrize("odppa_enabled", [False, True])
def test_active_odppa_statistics_cannot_use_a_conflicting_class_mapping(tmp_path, odppa_enabled):
    cfg = _cfg()
    cfg["quantification"].update(od_class=2, ppa_class=1)
    cfg["eval"]["odppa"]["enabled"] = odppa_enabled
    if odppa_enabled:
        with pytest.raises(ValueError, match="class|mapping|OD/PPA|odppa"):
            ErrorMapWriter(cfg, tmp_path)
    else:
        writer = ErrorMapWriter(cfg, tmp_path)
        assert writer.class_ids == {"od": 2, "ppa": 1}


@pytest.mark.parametrize("value", [0, 1, None, "yes", "", []])
def test_save_error_fundus_is_a_strict_boolean_setting(value):
    with pytest.raises(ValueError):
        parse_error_fundus_flag(value)


def test_normalization_accepts_documented_modes_and_boolean_strings():
    assert normalize_error_mode(None) == "none"
    assert normalize_error_mode(True) == "both"
    assert normalize_error_mode(" cart ") == "cartesian"
    assert normalize_error_classes(" PPA ") == "ppa"
    assert parse_error_fundus_flag(" true ") is True
    assert parse_error_fundus_flag(" FALSE ") is False
    with pytest.raises(ValueError):
        normalize_error_classes("all_classes")


@pytest.mark.parametrize("setting,value", [("error_map_alpha", -0.1), ("error_map_alpha", 1.1),
                                            ("error_map_alpha", float("nan")),
                                            ("error_map_source", "probabilities")])
def test_invalid_active_parameters_fail_before_creating_outputs(tmp_path, setting, value):
    cfg = _cfg()
    cfg["eval"][setting] = value
    with pytest.raises(ValueError):
        ErrorMapWriter(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()
