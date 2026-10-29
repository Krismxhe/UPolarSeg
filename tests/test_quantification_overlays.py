"""Independent numeric and coordinate contracts for quantification overlays.

The oracles below use periodic scalar interpolation and Cartesian trigonometry,
not the writer's geometry helpers. Tests exercise real readouts, PNGs and JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from src.evaluation.prediction_masks import PredictionMaskWriter
from src.evaluation.quantification_overlays import QuantificationOverlayWriter
from src.quantification.polar_readout import PolarReadout


PATH_FIELDS = {
    "overlay_polar_path", "overlay_cartesian_path", "overlay_metadata_path",
    "overlay_polar_hard_path", "overlay_cartesian_hard_path",
    "overlay_polar_soft_path", "overlay_cartesian_soft_path",
    "overlay_gt_polar_path", "overlay_gt_cartesian_path",
    "overlay_fundus_path", "overlay_gt_fundus_path",
}
RADII = [10, 9, 6, 4, 1, 8, 6, 4]
WIDTHS = [0, 1, 2, 5, 0, 2, 0, 1]


def _cfg(*, save_mode="both", dv_mode="max_center_chord", angle_bins=8,
         vertical_angle_deg=90.0, readouts="both", cartesian_size="input",
         layout="combined", gt_mode="none"):
    return {
        "train": {"img_size": 96},
        "dataset": {"num_classes": 3, "mask_mode": "index"},
        "polar": {
            "enabled": True,
            "geometry": {
                "mode": "linear", "angle_bins": angle_bins, "radius_bins": 32,
                "max_radius": 32.0, "center": [40.25, 49.5], "pt_type": "corner",
            },
        },
        "quantification": {
            "dv_mode": dv_mode, "vertical_angle_deg": vertical_angle_deg,
            "od_class": 1, "ppa_class": 2, "denominator_floor_px": 1.0,
        },
        "eval": {
            "save_quantification_overlays": save_mode,
            "quantification_overlay_readouts": readouts,
            "quantification_overlay_legend": False,
            "quantification_overlay_layout": layout,
            "save_quantification_gt_overlays": gt_mode,
            "prediction_cartesian_size": cartesian_size,
        },
    }


def _mask(radii=RADII, widths=WIDTHS):
    result = np.zeros((len(radii), 32), dtype=np.uint8)
    for angle, (radius, width) in enumerate(zip(radii, widths, strict=True)):
        result[angle, : radius + 1] = 1
        result[angle, radius + 1 : radius + width + 1] = 2
    return result


def _single(result):
    return {key: value[0] for key, value in result.items()}


def _measure(cfg, mask, probabilities=None):
    geometry = cfg["polar"]["geometry"]
    quant = cfg["quantification"]
    readout = PolarReadout(
        torch.ones(geometry["angle_bins"], geometry["radius_bins"], dtype=torch.bool),
        max_radius=geometry["max_radius"],
        dv_mode=quant["dv_mode"],
        vertical_angle_deg=quant["vertical_angle_deg"],
        denominator_floor_px=quant["denominator_floor_px"],
    )
    readout.geometry_metadata = {
        "img_size": cfg["train"]["img_size"], "center": tuple(geometry["center"]),
        "mode": "linear", "max_radius": geometry["max_radius"],
        "angle_bins": geometry["angle_bins"], "radius_bins": geometry["radius_bins"],
    }
    labels = torch.from_numpy(mask.astype(np.int64, copy=True)).unsqueeze(0)
    hard = _single(readout.measure_mask(labels))
    if probabilities is None:
        probabilities = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
    soft = _single(readout.read_probabilities(probabilities))
    return readout, hard, soft


def _json(path):
    def reject_constant(value):
        pytest.fail(f"Overlay JSON must not contain nonstandard numeric token {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _save(tmp_path, cfg, mask=None, *, name="case", probabilities=None,
          original_size=None, prediction_source="seg_logits",
          groundtruth_mask=None, reference_complete=True):
    mask = _mask() if mask is None else mask
    readout, hard, soft = _measure(cfg, mask, probabilities)
    groundtruth_hard = None
    if groundtruth_mask is not None:
        labels = torch.from_numpy(groundtruth_mask.astype(np.int64, copy=True)).unsqueeze(0)
        groundtruth_hard = _single(readout.measure_mask(labels))
    writer = QuantificationOverlayWriter(cfg, tmp_path)
    paths = writer.save(
        mask, name, readout=readout, hard=hard, soft=soft,
        original_size=original_size, prediction_source=prediction_source,
        groundtruth_mask=groundtruth_mask, groundtruth_hard=groundtruth_hard,
        reference_complete=reference_complete,
    )
    metadata = _json(paths["overlay_metadata_path"]) if paths["overlay_metadata_path"] else None
    return paths, metadata, (readout, hard, soft)


def _periodic_radius(profile, angle_deg):
    position = (angle_deg % 360.0) * len(profile) / 360.0
    index = math.floor(position)
    fraction = position - index
    return (
        (1.0 - fraction) * profile[index % len(profile)]
        + fraction * profile[(index + 1) % len(profile)]
    )


def _maximum_pair_oracle(profile):
    # Evaluate both sets of knots of r(theta) + r(theta + pi). The antipodal
    # knots matter to the piecewise-linear definition even for odd bin counts.
    angles = {360.0 * index / len(profile) for index in range(len(profile))}
    angles.update((angle + 180.0) % 360.0 for angle in tuple(angles))
    return max(
        _periodic_radius(profile, angle) + _periodic_radius(profile, angle + 180.0)
        for angle in angles
    )


def _point(center, radius, angle_deg):
    theta = math.radians(angle_deg)
    return np.asarray(center) + radius * np.array([math.cos(theta), math.sin(theta)])


def _assert_geometry(metadata, readout_name):
    """Check independent endpoint equations, distances and centre containment."""
    result = metadata["readouts"][readout_name]
    center = np.asarray(metadata["center_xy"], dtype=float)
    dr = float(metadata["radial_step"])
    angle_bins = metadata["polar_shape"][0]

    pmw = result["pmw"]
    inner, outer = pmw["radii"]
    theta = pmw["angle_deg"]
    endpoints = np.asarray(pmw["cartesian_endpoints"], dtype=float)
    expected = np.asarray([_point(center, inner, theta), _point(center, outer, theta)])
    np.testing.assert_allclose(endpoints, expected, atol=1e-6, rtol=0)
    assert np.linalg.norm(endpoints[1] - endpoints[0]) == pytest.approx(pmw["value"], abs=1e-6)
    assert outer - inner == pytest.approx(pmw["value"], abs=1e-6)
    np.testing.assert_allclose(
        pmw["polar_endpoints"],
        [[inner / dr, theta % 360 * angle_bins / 360],
         [outer / dr, theta % 360 * angle_bins / 360]],
        atol=1e-6, rtol=0,
    )

    directions = {}
    for key in ("dv", "dh"):
        item = result[key]
        radius1, radius2 = item["radii"]
        theta = item["angle_deg"]
        endpoints = np.asarray(item["cartesian_endpoints"], dtype=float)
        expected = np.asarray([
            _point(center, radius1, theta), _point(center, radius2, theta + 180.0),
        ])
        np.testing.assert_allclose(endpoints, expected, atol=1e-6, rtol=0)
        # Readout scalars are float32; JSON endpoint arithmetic uses doubles.
        assert radius1 + radius2 == pytest.approx(item["value"], abs=1e-5)
        assert np.linalg.norm(endpoints[1] - endpoints[0]) == pytest.approx(item["value"], abs=1e-5)
        if item["value"] > 0:
            centre_on_chord = endpoints[0] + radius1 / (radius1 + radius2) * (endpoints[1] - endpoints[0])
            np.testing.assert_allclose(centre_on_chord, center, atol=1e-6, rtol=0)
        rows = [theta % 360 * angle_bins / 360, (theta + 180.0) % 360 * angle_bins / 360]
        np.testing.assert_allclose(
            item["polar_segments"],
            [[[0.0, rows[0]], [radius1 / dr, rows[0]]],
             [[0.0, rows[1]], [radius2 / dr, rows[1]]]],
            atol=1e-6, rtol=0,
        )
        directions[key] = endpoints[1] - endpoints[0]
    assert np.dot(directions["dv"], directions["dh"]) == pytest.approx(0, abs=1e-6)
    assert result["dh"]["angle_deg"] % 180 == pytest.approx((result["dv"]["angle_deg"] + 90) % 180)
    assert result["dh"]["interpretation"] == "chord_perpendicular_to_selected_dv"


@pytest.mark.parametrize("mode", ["none", "polar", "cartesian", "both"])
def test_four_save_modes_create_only_requested_overlays(tmp_path, mode):
    cfg = _cfg(save_mode=mode)
    root = tmp_path / "exports"
    paths, metadata, _ = _save(root, cfg)
    assert set(paths) == PATH_FIELDS
    for kind, requested, shape in (
        ("polar", mode in {"polar", "both"}, (8, 32)),
        ("cartesian", mode in {"cartesian", "both"}, (96, 96)),
    ):
        path = paths[f"overlay_{kind}_path"]
        assert bool(path) is requested
        assert (root / f"overlays_{kind}").exists() is requested
        if requested:
            with Image.open(path) as picture:
                assert picture.mode == "RGB"
                assert picture.size == (shape[1], shape[0])
            assert metadata["output_shapes"][kind] == list(shape)
    if mode == "none":
        assert paths == dict.fromkeys(PATH_FIELDS, "")
        assert not root.exists()
    else:
        assert Path(paths["overlay_metadata_path"]).parent == root / "overlay_metadata"
        assert metadata["schema_version"] == "quantification-overlay-v1"
        assert metadata["prediction_source"] == "seg_logits"
        assert metadata["units"] == "resized_roi_px"
        assert metadata["input_shape"] == [8, 32]
        assert metadata["polar_shape"] == [8, 32]
        assert metadata["cartesian_input_shape"] == [96, 96]
        assert metadata["cartesian_shape"] == ([96, 96] if mode in {"cartesian", "both"} else None)


def test_none_is_a_noop_even_without_a_valid_case_or_readout(tmp_path):
    root = tmp_path / "disabled"
    writer = QuantificationOverlayWriter({}, root, mode="none")
    assert writer.mode == "none"
    paths = writer.save(
        None, "../unused", readout=None, hard=None, soft=None,
        original_size="not-a-size", prediction_source=None,
    )
    assert paths == dict.fromkeys(PATH_FIELDS, "")
    assert not root.exists()


@pytest.mark.parametrize("mode, axis, expected_dv", [
    ("fixed_vertical", 90.0, 12.0),
    ("fixed_vertical", 101.25, 11.0),
    ("max_center_chord", 90.0, 17.0),
])
def test_dv_and_pmw_match_readouts_and_independent_endpoints(tmp_path, mode, axis, expected_dv):
    cfg = _cfg(dv_mode=mode, vertical_angle_deg=axis)
    _, metadata, (_, hard, soft) = _save(tmp_path, cfg)
    assert metadata["dv_mode"] == mode
    assert metadata["center_xy"] == [40.25, 49.5]  # deliberately not the image centre
    assert metadata["radial_step"] == 1.0
    for name, numerical in (("hard", hard), ("soft", soft)):
        row = metadata["readouts"][name]
        assert row["status"] == "valid"
        assert row["valid"] is True
        assert row["raw"]["dv"] == pytest.approx(expected_dv)
        assert row["dv"]["value"] == pytest.approx(float(numerical["dv"]))
        assert row["pmw"]["value"] == 5.0
        assert row["pmw"]["radii"] == [4.0, 9.0]
        assert row["pmw"]["angle_deg"] == 135.0
        assert row["raw"]["pvdr"] == pytest.approx(5.0 / expected_dv)
        if mode == "max_center_chord":
            assert row["dv"]["angle_deg"] % 180.0 == 45.0
            assert sorted(row["dv"]["radii"]) == [8.0, 9.0]
        else:
            assert row["dv"]["angle_deg"] % 180.0 == axis % 180.0
        _assert_geometry(metadata, name)


def test_fixed_and_maximum_chord_produce_different_drawn_lines(tmp_path):
    fixed, _, _ = _save(tmp_path / "fixed", _cfg(dv_mode="fixed_vertical", readouts="hard"))
    maximum, _, _ = _save(tmp_path / "maximum", _cfg(readouts="hard"))
    for key in ("overlay_polar_path", "overlay_cartesian_path"):
        with Image.open(fixed[key]) as left, Image.open(maximum[key]) as right:
            assert not np.array_equal(np.asarray(left), np.asarray(right))


def test_odd_angle_bins_use_true_interpolated_antipodes(tmp_path):
    radii = [1, 2, 10, 3, 4]
    cfg = _cfg(angle_bins=5)
    _, metadata, _ = _save(tmp_path, cfg, _mask(radii, [2] * 5))
    assert _maximum_pair_oracle(radii) == 12.5
    for name in ("hard", "soft"):
        dv = metadata["readouts"][name]["dv"]
        assert dv["value"] == pytest.approx(_maximum_pair_oracle(radii))
        assert dv["angle_deg"] % 180.0 == pytest.approx(144.0)
        assert sorted(dv["radii"]) == [2.5, 10.0]
        _assert_geometry(metadata, name)


def test_radial_bin_coordinates_are_not_confused_with_pixel_lengths(tmp_path):
    cfg = _cfg()
    cfg["polar"]["geometry"]["max_radius"] = 16.0
    _, metadata, _ = _save(tmp_path, cfg)
    assert metadata["radial_step"] == 0.5
    for name in ("hard", "soft"):
        row = metadata["readouts"][name]
        assert row["dv"]["value"] == 8.5
        assert row["pmw"]["value"] == 2.5
        assert row["pmw"]["radii"] == [2.0, 4.5]
        np.testing.assert_allclose(row["pmw"]["polar_endpoints"], [[4, 3], [9, 3]])
        _assert_geometry(metadata, name)


def test_soft_geometry_is_labelled_as_equivalent_occupation_not_hard_boundary(tmp_path):
    cfg = _cfg()
    labels = torch.from_numpy(_mask().astype(np.int64)).unsqueeze(0)
    one_hot = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
    probabilities = one_hot * 0.8 + 0.2 / 3.0
    assert torch.equal(probabilities.argmax(dim=1), labels)
    _, metadata, _ = _save(tmp_path, cfg, probabilities=probabilities)
    soft = metadata["readouts"]["soft"]
    hard = metadata["readouts"]["hard"]
    background_occupation = 31 * (0.2 / 3.0)
    assert soft["interpretation"] == "equivalent_radial_occupation"
    assert soft["pmw"]["value"] == pytest.approx(0.8 * 5 + background_occupation, abs=1e-5)
    assert soft["pmw"]["radii"] == pytest.approx(
        [0.8 * 4 + background_occupation, 0.8 * 9 + 2 * background_occupation], abs=1e-5,
    )
    assert soft["dv"]["value"] == pytest.approx(0.8 * 17 + 2 * background_occupation, abs=1e-5)
    assert hard["pmw"]["radii"] == [4.0, 9.0]
    assert soft["pmw"]["radii"] != hard["pmw"]["radii"]
    _assert_geometry(metadata, "soft")


def test_original_size_uses_pixel_centre_mapping_under_unequal_axis_scaling(tmp_path):
    cfg = _cfg(cartesian_size="original")
    original_size = (144, 192)
    paths, metadata, _ = _save(tmp_path, cfg, original_size=original_size)
    assert metadata["cartesian_shape"] == list(original_size)
    with Image.open(paths["overlay_cartesian_path"]) as image:
        assert image.size == (192, 144)
    scale = np.array([2.0, 1.5])  # width then height
    for name in ("hard", "soft"):
        for measurement in ("pmw", "dv", "dh"):
            item = metadata["readouts"][name][measurement]
            roi = np.asarray(item["cartesian_endpoints"])
            expected = (roi + 0.5) * scale - 0.5
            actual = np.asarray(item["cartesian_output_endpoints"])
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        dv = metadata["readouts"][name]["dv"]
        # Original-image drawing distance changes anisotropically. Reported Dv
        # retains the input ROI's pixel unit, and must not silently be rescaled.
        output_length = np.linalg.norm(np.diff(dv["cartesian_output_endpoints"], axis=0))
        assert output_length == pytest.approx(17 * math.sqrt((2.0 ** 2 + 1.5 ** 2) / 2), abs=1e-5)
        assert dv["value"] == 17.0
        assert metadata["readouts"][name]["raw"]["dv"] == 17.0
        # Anisotropic display scaling need not preserve the input-space angle.
        dv_vector = np.diff(dv["cartesian_output_endpoints"], axis=0).ravel()
        dh_vector = np.diff(metadata["readouts"][name]["dh"]["cartesian_output_endpoints"], axis=0).ravel()
        assert abs(np.dot(dv_vector, dh_vector)) > 1.0


def test_polar_only_export_needs_no_original_size(tmp_path):
    cfg = _cfg(save_mode="polar", cartesian_size="original")
    _, metadata, _ = _save(tmp_path, cfg)
    for name in ("hard", "soft"):
        for measurement in ("pmw", "dv", "dh"):
            assert metadata["readouts"][name][measurement]["cartesian_output_endpoints"] is None


@pytest.mark.parametrize("selection, expected", [
    ("hard", {"hard"}), ("soft", {"soft"}), ("both", {"hard", "soft"}),
])
def test_readout_selection_is_explicit_in_metadata(tmp_path, selection, expected):
    _, metadata, _ = _save(tmp_path, _cfg(readouts=selection))
    assert set(metadata["readouts"]) == expected


def test_overlay_export_does_not_change_input_or_saved_class_index_masks(tmp_path):
    cfg = _cfg()
    mask = _mask()
    original = mask.copy()
    mask_paths = PredictionMaskWriter(cfg, tmp_path, mode="both").save(mask, "same_case")
    before = {path: Path(path).read_bytes() for path in set(mask_paths.values()) if path}
    _save(tmp_path, cfg, mask, name="same_case")
    np.testing.assert_array_equal(mask, original)
    for path, payload in before.items():
        assert Path(path).read_bytes() == payload
    with Image.open(mask_paths["pred_polar_path"]) as picture:
        assert picture.mode == "L"
        np.testing.assert_array_equal(np.asarray(picture), original)


def test_invalid_hard_geometry_is_flagged_and_not_drawn_as_valid(tmp_path):
    cfg = _cfg(readouts="hard")
    mask = _mask()
    mask[0, 3] = 0  # OD re-entry makes this hard measurement invalid.
    paths, metadata, (readout, hard, soft) = _save(tmp_path / "one", cfg, mask)
    row = metadata["readouts"]["hard"]
    assert row["status"] == "invalid"
    assert row["valid"] is False
    assert row["flags"]["reference_valid"] is False
    assert row["raw"]["dv"] == 17.0  # protected numeric output is still available
    # Finite candidates remain available for debugging, but the parent validity
    # flag prohibits drawing any of their lines as an accepted measurement.
    assert row["pmw"]["value"] == 5.0
    assert row["dv"]["value"] == 17.0
    assert row["dh"]["value"] == 8.0

    changed = {key: value.clone() for key, value in hard.items()}
    for key in ("pmw", "dv", "od_profile", "width_profile"):
        changed[key] = changed[key] * 0.5
    other_paths = QuantificationOverlayWriter(cfg, tmp_path / "two").save(
        mask, "case", readout=readout, hard=changed, soft=soft,
    )
    for key in ("overlay_polar_path", "overlay_cartesian_path"):
        with Image.open(paths[key]) as first, Image.open(other_paths[key]) as second:
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_soft_denominator_floor_is_explicit_and_keeps_raw_diameter(tmp_path):
    cfg = _cfg(readouts="soft")
    mask = np.zeros((8, 32), dtype=np.uint8)
    probabilities = torch.zeros(1, 3, 8, 32)
    probabilities[:, 1] = 0.001
    probabilities[:, 2] = 0.03
    probabilities[:, 0] = 0.969
    _, metadata, _ = _save(tmp_path, cfg, mask, probabilities=probabilities)
    row = metadata["readouts"]["soft"]
    assert row["status"] == "valid"
    assert row["valid"] is True  # finite soft surrogate, not a hard validity claim
    assert row["interpretation"] == "equivalent_radial_occupation"
    assert row["dv_floor_applied"] is True
    assert row["raw"]["dv"] == pytest.approx(2 * 31 * 0.001, abs=1e-6)
    assert row["raw"]["pvdr"] == pytest.approx(31 * 0.03, abs=1e-6)
    assert row["dv"]["value"] == pytest.approx(row["raw"]["dv"])


def test_nonfinite_readout_has_strict_json_nulls_and_no_valid_lines(tmp_path):
    cfg = _cfg(readouts="soft")
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    soft["pmw"] = torch.tensor(float("nan"))
    soft["dv"] = torch.tensor(float("inf"))
    soft["pvdr"] = torch.tensor(float("-inf"))
    soft["width_profile"][3] = float("nan")
    paths = QuantificationOverlayWriter(cfg, tmp_path).save(
        mask, "nonfinite", readout=readout, hard=hard, soft=soft,
    )
    row = _json(paths["overlay_metadata_path"])["readouts"]["soft"]
    assert row["status"] == "nonfinite"
    assert row["valid"] is False
    assert row["raw"] == {"pmw": None, "dv": None, "pvdr": None}
    assert row.get("pmw") is None
    assert row.get("dv") is None
    assert row.get("dh") is None


@pytest.mark.parametrize("source", ["seg_logits", "quant_logits"])
def test_supported_prediction_source_is_recorded_verbatim(tmp_path, source):
    _, metadata, _ = _save(tmp_path, _cfg(), prediction_source=source)
    assert metadata["prediction_source"] == source


@pytest.mark.parametrize("source", ["", "unknown_logits", None, 123])
def test_invalid_prediction_source_is_rejected(tmp_path, source):
    cfg = _cfg()
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    root = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError), match="source"):
        QuantificationOverlayWriter(cfg, root).save(
            mask, "case", readout=readout, hard=hard, soft=soft, prediction_source=source,
        )
    assert not list(root.rglob("*.png"))
    assert not list(root.rglob("*.json"))


@pytest.mark.parametrize("kind", ["mask_shape", "hard_profile", "soft_profile", "readout_geometry"])
def test_mismatched_prediction_or_readout_geometry_fails_without_outputs(tmp_path, kind):
    cfg = _cfg()
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    if kind == "mask_shape":
        mask = mask[:-1]
    elif kind == "hard_profile":
        hard["od_profile"] = hard["od_profile"][:-1]
    elif kind == "soft_profile":
        soft["width_profile"] = soft["width_profile"][:-1]
    else:
        cfg["polar"]["geometry"]["max_radius"] = 16.0
    root = tmp_path / "invalid"
    with pytest.raises((ValueError, TypeError)):
        QuantificationOverlayWriter(cfg, root).save(
            mask, "case", readout=readout, hard=hard, soft=soft,
        )
    assert not list(root.rglob("*.png"))
    assert not list(root.rglob("*.json"))


@pytest.mark.parametrize("unsupported", ["cartesian_input", "log_polar"])
def test_unsupported_input_spaces_are_rejected(tmp_path, unsupported):
    cfg = _cfg()
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    if unsupported == "cartesian_input":
        cfg["polar"]["enabled"] = False
    else:
        cfg["polar"]["geometry"]["mode"] = "log"
    with pytest.raises(ValueError):
        QuantificationOverlayWriter(cfg, tmp_path / "invalid").save(
            mask, "case", readout=readout, hard=hard, soft=soft,
        )


def test_cartesian_original_size_requires_explicit_dimensions(tmp_path):
    cfg = _cfg(cartesian_size="original")
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    with pytest.raises(ValueError, match="original_size"):
        QuantificationOverlayWriter(cfg, tmp_path).save(
            mask, "case", readout=readout, hard=hard, soft=soft,
        )


def test_custom_output_directories_preserve_independent_path_contract(tmp_path):
    cfg = _cfg()
    mask = _mask()
    readout, hard, soft = _measure(cfg, mask)
    roots = {
        "polar_dir": tmp_path / "polar_display",
        "cartesian_dir": tmp_path / "cart_display",
        "metadata_dir": tmp_path / "geometry_records",
    }
    paths = QuantificationOverlayWriter(cfg, tmp_path / "unused_default", **roots).save(
        mask, "custom", readout=readout, hard=hard, soft=soft,
    )
    assert Path(paths["overlay_polar_path"]).parent == roots["polar_dir"]
    assert Path(paths["overlay_cartesian_path"]).parent == roots["cartesian_dir"]
    assert Path(paths["overlay_metadata_path"]).parent == roots["metadata_dir"]
    assert not (tmp_path / "unused_default").exists()


@pytest.mark.parametrize("radii, expected_dv, expected_dh, expected_axis", [
    (RADII, 17.0, 8.0, 45.0),
    ([1, 2, 10, 3, 4], 12.5, 5.0, 144.0),
    ([3, 4, 9, 5, 2, 7], 16.0, 7.0, 120.0),
])
def test_dh_is_perpendicular_to_selected_dv_with_quarter_bin_interpolation(
    tmp_path, radii, expected_dv, expected_dh, expected_axis,
):
    cfg = _cfg(angle_bins=len(radii))
    _, metadata, (_, hard, soft) = _save(tmp_path, cfg, _mask(radii, [2] * len(radii)))
    for name, numerical in (("hard", hard), ("soft", soft)):
        row = metadata["readouts"][name]
        axis = (expected_axis + 90.0) % 180.0
        independent = _periodic_radius(radii, axis) + _periodic_radius(radii, axis + 180.0)
        assert row["dv"]["value"] == expected_dv
        assert row["dh"]["angle_deg"] == pytest.approx(axis)
        assert row["dh"]["value"] == pytest.approx(independent)
        assert row["dh"]["value"] == pytest.approx(expected_dh)
        assert row["dh"]["value"] != row["dv"]["value"]
        assert "dh" not in numerical  # derived drawing, not a new readout/loss metric
        assert set(row["raw"]) == {"pmw", "dv", "pvdr"}
        _assert_geometry(metadata, name)


@pytest.mark.parametrize("radii", [RADII, [1, 2, 10, 3, 4]])
def test_dv_and_dh_rotate_together_under_circular_angle_shift(tmp_path, radii):
    count = len(radii)
    cfg = _cfg(angle_bins=count)
    mask = _mask(radii, [2] * count)
    _, original, _ = _save(tmp_path / "original", cfg, mask)
    _, rotated, _ = _save(tmp_path / "rotated", cfg, np.roll(mask, 1, axis=0).copy())
    for name in ("hard", "soft"):
        for key in ("dv", "dh"):
            old = original["readouts"][name][key]
            new = rotated["readouts"][name][key]
            assert new["value"] == pytest.approx(old["value"])
            assert new["angle_deg"] % 180 == pytest.approx((old["angle_deg"] + 360.0 / count) % 180)
        _assert_geometry(rotated, name)


def test_soft_dh_uses_soft_dv_winner_independently_of_the_hard_winner(tmp_path):
    cfg = _cfg()
    mask = _mask()
    labels = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
    one_hot = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
    probabilities = one_hot * 0.98 + 0.02 / 3.0
    # Both uncertain rays retain exactly the same argmax mask, but accumulate
    # substantial OD occupation outside the discrete OD boundary.
    for angle in (0, 4):
        for radial in range(32):
            if mask[angle, radial] == 1:
                probabilities[0, :, angle, radial] = torch.tensor([0.1, 0.8, 0.1])
            else:
                probabilities[0, :, angle, radial] = torch.tensor([0.5, 0.49, 0.01])
    assert torch.equal(probabilities.argmax(dim=1), labels)
    _, metadata, _ = _save(tmp_path, cfg, mask, probabilities=probabilities)
    hard, soft = metadata["readouts"]["hard"], metadata["readouts"]["soft"]
    assert hard["dv"]["angle_deg"] % 180 == 45.0
    assert hard["dh"]["angle_deg"] % 180 == 135.0
    assert soft["dv"]["angle_deg"] % 180 == 0.0
    assert soft["dh"]["angle_deg"] % 180 == 90.0
    _assert_geometry(metadata, "hard")
    _assert_geometry(metadata, "soft")


def test_combined_and_separate_exports_have_identical_geometry_and_distinct_images(tmp_path):
    cfg = _cfg(layout="combined")
    combined_paths, combined_metadata, _ = _save(tmp_path, cfg, name="shared")
    combined_images = {
        key: Path(combined_paths[key]).read_bytes()
        for key in ("overlay_polar_path", "overlay_cartesian_path")
    }
    cfg["eval"]["quantification_overlay_layout"] = "separate"
    separate_paths, separate_metadata, _ = _save(tmp_path, cfg, name="shared")
    assert combined_metadata["layout"] == "combined"
    assert separate_metadata["layout"] == "separate"
    assert separate_metadata["readouts"] == combined_metadata["readouts"]
    assert separate_paths["overlay_polar_path"] == ""
    assert separate_paths["overlay_cartesian_path"] == ""
    for space in ("polar", "cartesian"):
        for readout_name in ("hard", "soft"):
            key = f"overlay_{space}_{readout_name}_path"
            path = Path(separate_paths[key])
            assert path.is_file()
            assert path.parent == tmp_path / f"overlays_{space}" / readout_name
            assert str(path) not in combined_paths.values()
            assert combined_paths[key] == ""
    for key, previous in combined_images.items():
        assert Path(combined_paths[key]).read_bytes() == previous


@pytest.mark.parametrize("selection", ["hard", "soft"])
def test_separate_layout_writes_only_selected_readout(tmp_path, selection):
    cfg = _cfg(layout="separate", readouts=selection)
    paths, metadata, _ = _save(tmp_path, cfg)
    assert set(metadata["readouts"]) == {selection}
    for space in ("polar", "cartesian"):
        assert paths[f"overlay_{space}_path"] == ""
        assert paths[f"overlay_{space}_{selection}_path"]
        other = "soft" if selection == "hard" else "hard"
        assert paths[f"overlay_{space}_{other}_path"] == ""
        assert not (tmp_path / f"overlays_{space}" / other).exists()


def _groundtruth():
    return _mask([5, 4, 7, 3, 2, 5, 6, 4], [1, 2, 4, 0, 1, 1, 0, 1])


def test_groundtruth_has_its_own_mask_measurements_and_files(tmp_path):
    cfg = _cfg(gt_mode="both")
    truth = _groundtruth()
    original = truth.copy()
    paths, metadata, _ = _save(tmp_path, cfg, groundtruth_mask=truth)
    groundtruth = metadata["groundtruth"]
    assert groundtruth["source"] == "ground_truth"
    assert groundtruth["annotation_complete"] is True
    assert "soft" not in groundtruth
    target = groundtruth["hard"]
    predicted = metadata["readouts"]["hard"]
    assert target["raw"]["dv"] == 13.0
    assert target["raw"]["pmw"] == 4.0
    assert target["dh"]["value"] == 7.0
    assert predicted["raw"]["dv"] == 17.0
    assert predicted["raw"]["pmw"] == 5.0
    for space in ("polar", "cartesian"):
        gt_path = paths[f"overlay_gt_{space}_path"]
        pred_path = paths[f"overlay_{space}_path"]
        assert gt_path and gt_path != pred_path
        with Image.open(gt_path) as gt_image, Image.open(pred_path) as pred_image:
            assert not np.array_equal(np.asarray(gt_image), np.asarray(pred_image))
    np.testing.assert_array_equal(truth, original)
    _assert_geometry({**metadata, "readouts": {"target": target}}, "target")


@pytest.mark.parametrize("gt_mode", ["polar", "cartesian", "both"])
def test_gt_only_export_does_not_require_prediction_data(tmp_path, gt_mode):
    cfg = _cfg(save_mode="none", gt_mode=gt_mode, cartesian_size="original")
    truth = _groundtruth()
    readout, hard, _ = _measure(cfg, truth)
    writer = QuantificationOverlayWriter(cfg, tmp_path)
    original_size = (144, 192) if gt_mode in {"cartesian", "both"} else None
    paths = writer.save(
        None, "target", readout=readout, hard=None, soft=None,
        groundtruth_mask=truth, groundtruth_hard=hard,
        original_size=original_size,
    )
    metadata = _json(paths["overlay_metadata_path"])
    assert writer.mode == "none" and writer.gt_mode == gt_mode
    assert metadata["readouts"] == {}
    assert metadata["output_shapes"] == {"polar": None, "cartesian": None}
    assert metadata["groundtruth"]["hard"]["dv"]["value"] == 13.0
    for key in PATH_FIELDS - {"overlay_metadata_path", "overlay_gt_polar_path", "overlay_gt_cartesian_path"}:
        assert paths[key] == ""
    for space, requested in (
        ("polar", gt_mode in {"polar", "both"}),
        ("cartesian", gt_mode in {"cartesian", "both"}),
    ):
        path = paths[f"overlay_gt_{space}_path"]
        assert bool(path) is requested
        if requested:
            shape = (8, 32) if space == "polar" else original_size
            with Image.open(path) as picture:
                assert picture.size == (shape[1], shape[0])
            assert metadata["gt_output_shapes"][space] == list(shape)
    if original_size is not None:
        for key in ("pmw", "dv", "dh"):
            item = metadata["groundtruth"]["hard"][key]
            expected = (np.asarray(item["cartesian_endpoints"]) + 0.5) * [2.0, 1.5] - 0.5
            np.testing.assert_allclose(item["cartesian_output_endpoints"], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("reference_complete, expected", [
    (False, False), ("false", False), ("0", False),
    (True, True), ("true", True), ("1", True),
])
def test_gt_annotation_completeness_distinguishes_unknown_ppa_from_known_od(
    tmp_path, reference_complete, expected,
):
    cfg = _cfg(gt_mode="both")
    _, metadata, _ = _save(
        tmp_path, cfg, groundtruth_mask=_groundtruth(), reference_complete=reference_complete,
    )
    target = metadata["groundtruth"]
    assert target["annotation_complete"] is expected
    row = target["hard"]
    assert row["valid"] is True
    assert row["dv"]["value"] == 13.0
    assert row["dh"]["value"] == 7.0
    assert row["dv"]["annotation_valid"] is True
    assert row["dh"]["annotation_valid"] is True
    assert row["pmw"]["annotation_valid"] is expected
    if not expected:
        assert row["pmw"]["status"] == "unannotated"


@pytest.mark.parametrize("missing", ["mask", "readout", "both"])
def test_requested_gt_export_rejects_missing_target_inputs(tmp_path, missing):
    cfg = _cfg(gt_mode="polar")
    mask, truth = _mask(), _groundtruth()
    readout, hard, soft = _measure(cfg, mask)
    _, gt_hard, _ = _measure(cfg, truth)
    with pytest.raises((ValueError, TypeError), match="groundtruth|ground.truth|GT"):
        QuantificationOverlayWriter(cfg, tmp_path).save(
            mask, "missing", readout=readout, hard=hard, soft=soft,
            groundtruth_mask=None if missing in {"mask", "both"} else truth,
            groundtruth_hard=None if missing in {"readout", "both"} else gt_hard,
        )
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("mode,gt_mode,size,expected", [
    ("none", "none", "original", False),
    ("polar", "polar", "original", False),
    ("none", "cartesian", "original", True),
    ("polar", "both", "original", True),
    ("cartesian", "none", "original", True),
    ("both", "both", "input", False),
])
def test_original_size_requirement_includes_gt_exports(tmp_path, mode, gt_mode, size, expected):
    cfg = _cfg(save_mode=mode, gt_mode=gt_mode, cartesian_size=size)
    writer = QuantificationOverlayWriter(cfg, tmp_path)
    assert writer.needs_original_size is expected
