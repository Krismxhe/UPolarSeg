"""Fundus exports: real RGB files, hard readouts and independent coordinates.

These tests deliberately use an off-centre polar origin and a non-square
original image. They do not import renderer helpers as numeric oracles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from src.evaluation.quantification_overlays import (
    QuantificationOverlayWriter,
    normalize_fundus_mode,
)
from src.quantification.polar_readout import PolarReadout


RADII = [24, 19, 12, 10, 10, 17, 12, 10, 4, 16, 13, 9, 10, 8, 11, 9]
WIDTHS = [0, 1, 2, 10, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
OLD_IMAGE_FIELDS = {
    "overlay_polar_path", "overlay_cartesian_path",
    "overlay_polar_hard_path", "overlay_cartesian_hard_path",
    "overlay_polar_soft_path", "overlay_cartesian_soft_path",
    "overlay_gt_polar_path", "overlay_gt_cartesian_path",
}
FUNDUS_FIELDS = {"overlay_fundus_path", "overlay_gt_fundus_path"}


def _cfg(*, mode="prediction", size="original", alpha=0.0, dv_mode="max_center_chord"):
    return {
        "train": {"img_size": 160},
        "dataset": {"num_classes": 3, "mask_mode": "index"},
        "polar": {"enabled": True, "geometry": {
            "center": [61.25, 89.5], "max_radius": 64.0,
            "pt_type": "corner", "radius_bins": 64, "angle_bins": 16,
            "mode": "linear",
        }},
        "quantification": {
            "dv_mode": dv_mode, "vertical_angle_deg": 90.0,
            "od_class": 1, "ppa_class": 2, "denominator_floor_px": 1.0,
        },
        "eval": {
            "save_fundus_overlays": mode, "fundus_overlay_alpha": alpha,
            "save_predictions": "none", "save_quantification_overlays": "none",
            "save_quantification_gt_overlays": "none",
            # These settings must not make a fundus-only panel need soft data.
            "quantification_overlay_readouts": "soft",
            "quantification_overlay_layout": "separate",
            "quantification_overlay_legend": False,
            "prediction_cartesian_size": size,
        },
    }


def _mask(radii=RADII, widths=WIDTHS):
    mask = np.zeros((16, 64), dtype=np.uint8)
    for angle, (radius, width) in enumerate(zip(radii, widths, strict=True)):
        mask[angle, :radius + 1] = 1
        mask[angle, radius + 1:radius + width + 1] = 2
    return mask


def _groundtruth():
    widths = [2] * 16
    widths[3] = 16
    return _mask([14] * 16, widths)


def _readout(cfg):
    geometry = cfg["polar"]["geometry"]
    q = cfg["quantification"]
    readout = PolarReadout(
        torch.ones(16, 64, dtype=torch.bool), max_radius=64.0,
        dv_mode=q["dv_mode"], vertical_angle_deg=q["vertical_angle_deg"],
        denominator_floor_px=1.0,
    )
    readout.geometry_metadata = {
        "img_size": 160, "center": tuple(geometry["center"]),
        "mode": "linear", "max_radius": 64.0,
        "angle_bins": 16, "radius_bins": 64,
    }
    return readout


def _hard(readout, mask):
    result = readout.measure_mask(torch.from_numpy(mask.astype(np.int64)).unsqueeze(0))
    return {key: value[0] for key, value in result.items()}


def _image(tmp_path, shape=(240, 320)):
    y, x = np.indices(shape)
    rgb = np.stack((30 + x % 40, 70 + y % 60, 90 + (x + 2 * y) % 50), axis=-1).astype(np.uint8)
    path = tmp_path / "source_rgb.png"
    Image.fromarray(rgb).save(path)
    return path, rgb


def _load_json(path):
    def reject(token):
        pytest.fail(f"Nonstandard JSON numeric constant: {token}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _save(tmp_path, cfg, *, image_path, original_size=None, name="case",
          mask=None, groundtruth=None, hard=None, reference_complete=True, **writer_kwargs):
    readout = _readout(cfg)
    mode = cfg["eval"]["save_fundus_overlays"]
    prediction = (_mask() if mask is None else mask) if mode in {"prediction", "both"} else None
    hard = (_hard(readout, prediction) if hard is None else hard) if prediction is not None else None
    gt = (_groundtruth() if groundtruth is None else groundtruth) if mode in {"groundtruth", "both"} else None
    gt_hard = _hard(readout, gt) if gt is not None else None
    writer = QuantificationOverlayWriter(cfg, tmp_path, **writer_kwargs)
    paths = writer.save(
        prediction, name, readout=readout, hard=hard, soft=None,
        image_path=image_path, original_size=original_size,
        groundtruth_mask=gt, groundtruth_hard=gt_hard,
        reference_complete=reference_complete,
    )
    metadata = _load_json(paths["overlay_metadata_path"]) if paths["overlay_metadata_path"] else None
    return paths, metadata, writer


def _point(radius, angle_deg):
    theta = math.radians(angle_deg)
    return np.array([61.25 + radius * math.cos(theta), 89.5 + radius * math.sin(theta)])


def _mapped(points, output_shape):
    # Pixel-centre correspondence of resizing a 160 x 160 input image.
    return (np.asarray(points) + 0.5) * [output_shape[1] / 160, output_shape[0] / 160] - 0.5


def _cart_labels(mask, shape=(160, 160)):
    # Independent OpenCV reference with explicit geometry and nearest labels.
    cart = cv2.warpPolar(
        mask, (160, 160), (61.25, 89.5), 64.0,
        cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST,
    )
    return cv2.resize(cart, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


@pytest.mark.parametrize("mode", ["none", "prediction", "groundtruth", "both"])
@pytest.mark.parametrize("size", ["input", "original"])
def test_fundus_modes_are_independent_of_mask_panels_and_soft_selection(tmp_path, mode, size):
    source, rgb = _image(tmp_path)
    cfg = _cfg(mode=mode, size=size)
    root = tmp_path / "exports"
    paths, metadata, writer = _save(root, cfg, image_path=source)
    requested = {
        "overlay_fundus_path": mode in {"prediction", "both"},
        "overlay_gt_fundus_path": mode in {"groundtruth", "both"},
    }
    assert FUNDUS_FIELDS <= set(paths)
    assert all(paths[key] == "" for key in OLD_IMAGE_FIELDS)
    assert writer.fundus_mode == mode
    assert writer.needs_original_size is (mode != "none" and size == "original")
    output_shape = (160, 160) if size == "input" else rgb.shape[:2]
    for key, selected in requested.items():
        assert bool(paths[key]) is selected
        if selected:
            with Image.open(paths[key]) as exported:
                assert exported.mode == "RGB"
                assert (exported.height, exported.width) == output_shape
    if mode == "none":
        assert not root.exists()
        assert not any(paths.values())
        return
    assert metadata["readouts"] == {}
    assert metadata["groundtruth"] == {}
    fundus = metadata["fundus"]
    assert fundus["mode"] == mode
    assert fundus["original_shape"] == [240, 320]
    assert fundus["output_shape"] == list(output_shape)
    assert Path(fundus["source_image"]) == source
    assert fundus["alpha"] == 0.0
    assert fundus["units"] == "resized_roi_px"
    assert set(fundus["prediction"]) == ({"hard"} if requested["overlay_fundus_path"] else set())
    assert bool(fundus["groundtruth"]) is requested["overlay_gt_fundus_path"]
    assert not any((root / directory).exists() for directory in (
        "masks_polar", "masks_cartesian", "overlays_polar", "overlays_cartesian",
        "overlays_gt_polar", "overlays_gt_cartesian",
    ))


@pytest.mark.parametrize("dv_mode,dv_angle,dv_radii,dh_angle,dh_radii", [
    ("fixed_vertical", 90.0, (10.0, 10.0), 0.0, (24.0, 4.0)),
    ("max_center_chord", 22.5, (19.0, 16.0), 112.5, (17.0, 8.0)),
])
def test_original_non_square_geometry_preserves_roi_measurements_and_maps_endpoints(
    tmp_path, dv_mode, dv_angle, dv_radii, dh_angle, dh_radii,
):
    source, _ = _image(tmp_path)
    paths, metadata, _ = _save(tmp_path / "exports", _cfg(dv_mode=dv_mode), image_path=source)
    record = metadata["fundus"]["prediction"]["hard"]
    assert record["valid"] is True
    assert record["interpretation"] == "hard_radial_boundary"
    assert record["raw"]["pmw"] == 10.0
    assert record["raw"]["dv"] == sum(dv_radii)
    assert record["raw"]["pvdr"] == pytest.approx(10.0 / sum(dv_radii))
    expected_center = _mapped([61.25, 89.5], (240, 320))
    np.testing.assert_allclose(metadata["fundus"]["output_center_xy"], expected_center, atol=1e-6)
    vectors = {}
    for key, angle, radii in (("dv", dv_angle, dv_radii), ("dh", dh_angle, dh_radii)):
        item = record[key]
        expected = np.array([_point(radii[0], angle), _point(radii[1], angle + 180)])
        assert item["angle_deg"] == pytest.approx(angle)
        assert item["value"] == pytest.approx(sum(radii))
        np.testing.assert_allclose(item["cartesian_endpoints"], expected, atol=1e-6)
        endpoints = np.asarray(item["cartesian_output_endpoints"])
        np.testing.assert_allclose(endpoints, _mapped(expected, (240, 320)), atol=1e-6)
        fraction = radii[0] / sum(radii)
        np.testing.assert_allclose(endpoints[0] + fraction * (endpoints[1] - endpoints[0]), expected_center, atol=1e-6)
        vectors[key] = expected[1] - expected[0]
    assert np.dot(vectors["dv"], vectors["dh"]) == pytest.approx(0, abs=1e-6)
    pmw = record["pmw"]
    expected_pmw = np.array([_point(10, 67.5), _point(20, 67.5)])
    np.testing.assert_allclose(pmw["cartesian_endpoints"], expected_pmw, atol=1e-6)
    np.testing.assert_allclose(pmw["cartesian_output_endpoints"], _mapped(expected_pmw, (240, 320)), atol=1e-6)
    image = np.asarray(Image.open(paths["overlay_fundus_path"]))
    for item in (record["pmw"], record["dv"], record["dh"]):
        for x, y in item["cartesian_output_endpoints"]:
            patch = image[round(y)-2:round(y)+3, round(x)-2:round(x)+3]
            assert patch.size and np.any(patch.max(axis=-1) > 180), "A mapped measurement endpoint must be visible"


def test_alpha_zero_preserves_rgb_background_and_alpha_blends_only_class_regions(tmp_path):
    source, rgb = _image(tmp_path, (160, 160))
    original_bytes = source.read_bytes()
    # A broad regular OD leaves a substantial interior away from the contour,
    # centre marker and both diameters, independent of anti-aliasing details.
    mask = _mask([24] * 16, [8] * 16)
    before_mask = mask.copy()
    images = []
    for alpha in (0.0, 0.5, 1.0):
        paths, metadata, _ = _save(tmp_path / str(alpha), _cfg(size="input", alpha=alpha),
                                   image_path=source, mask=mask)
        images.append(np.asarray(Image.open(paths["overlay_fundus_path"])))
        assert metadata["fundus"]["alpha"] == alpha
        np.testing.assert_array_equal(images[-1][:32, :32], rgb[:32, :32])
    labels = _cart_labels(mask)
    interior = cv2.distanceTransform((labels == 1).astype(np.uint8), cv2.DIST_L2, 5) > 4
    # Uniform profiles choose horizontal/vertical diameter axes. Exclude a
    # geometric band around those lines, including anti-aliasing and the centre
    # marker, instead of selecting pixels according to rendered output values.
    y, x = np.indices(labels.shape)
    untouched = interior & (np.abs(x - 61.25) > 5) & (np.abs(y - 89.5) > 5)
    assert untouched.sum() > 30, "Alpha zero must retain fundus texture inside the OD"
    np.testing.assert_array_equal(images[0][untouched], rgb[untouched])
    assert np.count_nonzero(np.any(images[0] != rgb, axis=-1)) > 40, "Contours and measurements remain visible"
    assert np.all(np.any(images[1][untouched] != rgb[untouched], axis=-1))
    np.testing.assert_array_equal(images[2][untouched], np.broadcast_to([96, 165, 250], (untouched.sum(), 3)))
    np.testing.assert_allclose(images[1][untouched],
                               (images[0][untouched].astype(float) + images[2][untouched]) / 2,
                               atol=1.0, rtol=0)
    np.testing.assert_array_equal(mask, before_mask)
    assert source.read_bytes() == original_bytes


def test_od_ppa_shared_boundary_keeps_od_blue_and_ppa_outer_edge_stays_purple(tmp_path):
    source, _ = _image(tmp_path, (640, 640))
    cfg = _cfg()
    mask = _mask([24] * 16, [8] * 16)
    hard = _hard(_readout(cfg), mask)
    # Isolate class contours from measurement lines; the configured centre
    # marker is far inside the OD and cannot reach either tested boundary.
    hard["reference_valid"] = torch.tensor(False)
    paths, _, _ = _save(tmp_path / "exports", cfg, image_path=source, mask=mask, hard=hard)
    rendered = np.asarray(Image.open(paths["overlay_fundus_path"]))
    labels = _cart_labels(mask, (640, 640))
    od, ppa, background = (labels == value for value in (1, 2, 0))
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    shared = od & cv2.dilate(ppa.astype(np.uint8), cross).astype(bool)
    outer = ppa & cv2.dilate(background.astype(np.uint8), cross).astype(bool)

    def straight_edge_pixels(boundary):
        # Select centres of straight runs, avoiding diagonal/corner AA pixels.
        vertical = np.zeros_like(boundary)
        horizontal = np.zeros_like(boundary)
        vertical[1:-1] = boundary[:-2] & boundary[1:-1] & boundary[2:]
        horizontal[:, 1:-1] = boundary[:, :-2] & boundary[:, 1:-1] & boundary[:, 2:]
        return vertical | horizontal

    shared = straight_edge_pixels(shared)
    outer = straight_edge_pixels(outer)
    assert shared.sum() > 100 and outer.sum() > 100
    # At this output size the stroke has an opaque core. Reversing the intended
    # draw order lets the PPA hole contour overwrite the shared OD outline.
    od_blue = np.all(rendered == [96, 165, 250], axis=-1)
    ppa_purple = np.all(rendered == [232, 121, 249], axis=-1)
    assert od_blue[shared].mean() > 0.9, "PPA must not cover the shared OD outline"
    assert ppa_purple[outer].mean() > 0.9, "The outer PPA outline must remain purple"


def test_gt_uses_its_own_mask_and_hard_measurements_on_the_same_fundus(tmp_path):
    source, rgb = _image(tmp_path, (160, 160))
    paths, metadata, _ = _save(tmp_path / "exports", _cfg(mode="both", size="input"), image_path=source)
    fundus = metadata["fundus"]
    assert fundus["prediction"]["hard"]["raw"]["pmw"] == 10.0
    assert fundus["prediction"]["hard"]["raw"]["dv"] == 35.0
    assert fundus["groundtruth"]["hard"]["raw"]["pmw"] == 16.0
    assert fundus["groundtruth"]["hard"]["raw"]["dv"] == 28.0
    pred = np.asarray(Image.open(paths["overlay_fundus_path"]))
    gt = np.asarray(Image.open(paths["overlay_gt_fundus_path"]))
    assert paths["overlay_fundus_path"] != paths["overlay_gt_fundus_path"]
    assert not np.array_equal(pred, gt)
    np.testing.assert_array_equal(pred[:32, :32], rgb[:32, :32])
    np.testing.assert_array_equal(gt[:32, :32], rgb[:32, :32])


def test_partial_gt_skips_pmw_pixels_and_keeps_od_diameters(tmp_path):
    source, rgb = _image(tmp_path, (160, 160))
    cfg = _cfg(mode="groundtruth", size="input")
    complete_paths, _, _ = _save(tmp_path / "complete", cfg, image_path=source)
    partial_paths, metadata, _ = _save(tmp_path / "partial", cfg, image_path=source, reference_complete="false")
    gt = metadata["fundus"]["groundtruth"]
    assert gt["annotation_complete"] is False
    assert gt["hard"]["pmw"]["annotation_valid"] is False
    assert gt["hard"]["pmw"]["status"] == "unannotated"
    assert gt["hard"]["dv"]["annotation_valid"] is True
    assert gt["hard"]["dh"]["annotation_valid"] is True
    complete = np.asarray(Image.open(complete_paths["overlay_gt_fundus_path"]))
    partial = np.asarray(Image.open(partial_paths["overlay_gt_fundus_path"]))
    x, y = np.rint(_point(22, 67.5)).astype(int)
    assert np.any(complete[y-1:y+2, x-1:x+2] != partial[y-1:y+2, x-1:x+2])
    # Both diameter chords still cross the off-centre origin.
    cx, cy = (round(value) for value in (61.25, 89.5))
    np.testing.assert_array_equal(complete[cy-3:cy+4, cx-3:cx+4], partial[cy-3:cy+4, cx-3:cx+4])
    assert np.any(partial[cy-3:cy+4, cx-3:cx+4] != rgb[cy-3:cy+4, cx-3:cx+4])


def test_invalid_hard_candidates_cannot_draw_measurement_lines(tmp_path):
    source, _ = _image(tmp_path, (160, 160))
    cfg = _cfg(size="input")
    hard = _hard(_readout(cfg), _mask())
    hard["reference_valid"] = torch.tensor(False)
    first_paths, metadata, _ = _save(tmp_path / "first", cfg, image_path=source, hard=hard)
    record = metadata["fundus"]["prediction"]["hard"]
    assert record["valid"] is False
    assert record["status"] == "invalid"
    altered = dict(hard)
    for key in ("od_profile", "width_profile", "pmw", "dv"):
        altered[key] = hard[key] * 0.5
    second_paths, _, _ = _save(tmp_path / "second", cfg, image_path=source, hard=altered)
    # Underlying contours and configured centre are identical; only invalid
    # candidate measurement geometry differs, so the PNG pixels must agree.
    np.testing.assert_array_equal(np.asarray(Image.open(first_paths["overlay_fundus_path"])),
                                  np.asarray(Image.open(second_paths["overlay_fundus_path"])))


def test_soft_mask_panel_and_hard_fundus_keep_separate_readout_records(tmp_path):
    source, _ = _image(tmp_path)
    cfg = _cfg()
    cfg["eval"]["save_quantification_overlays"] = "polar"
    readout = _readout(cfg)
    mask = _mask()
    hard = _hard(readout, mask)
    labels = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
    probabilities = 0.8 * torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() + 0.2 / 3
    soft = {key: value[0] for key, value in readout.read_probabilities(probabilities).items()}
    paths = QuantificationOverlayWriter(cfg, tmp_path / "exports").save(
        mask, "case", readout=readout, hard=hard, soft=soft, image_path=source,
    )
    metadata = _load_json(paths["overlay_metadata_path"])
    assert set(metadata["readouts"]) == {"soft"}
    assert metadata["readouts"]["soft"]["interpretation"] == "equivalent_radial_occupation"
    assert set(metadata["fundus"]["prediction"]) == {"hard"}
    assert metadata["fundus"]["prediction"]["hard"]["raw"]["pmw"] == 10.0
    assert metadata["readouts"]["soft"]["raw"]["pmw"] != 10.0
    assert Path(paths["overlay_polar_soft_path"]).is_file()
    assert Path(paths["overlay_fundus_path"]).is_file()
    assert paths["overlay_polar_hard_path"] == ""


def test_disabled_fundus_never_opens_an_image_or_creates_outputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled fundus must not read the source image")
    monkeypatch.setattr(Image, "open", forbidden)
    writer = QuantificationOverlayWriter({}, tmp_path / "disabled")
    paths = writer.save(None, "../unused", readout=None, hard=None, soft=None,
                        image_path=object(), original_size="invalid", prediction_source=None)
    assert not any(paths.values())
    assert not (tmp_path / "disabled").exists()


def test_disabled_fundus_does_not_require_image_path_for_old_polar_exports(tmp_path, monkeypatch):
    cfg = _cfg(mode="none")
    cfg["eval"]["save_quantification_overlays"] = "polar"
    cfg["eval"]["quantification_overlay_readouts"] = "hard"
    readout = _readout(cfg)
    mask = _mask()
    def forbidden(*args, **kwargs):
        raise AssertionError("A polar mask overlay does not read fundus RGB")
    monkeypatch.setattr(Image, "open", forbidden)
    paths = QuantificationOverlayWriter(cfg, tmp_path / "exports").save(
        mask, "case", readout=readout, hard=_hard(readout, mask), soft=None,
        image_path=tmp_path / "missing.png",
    )
    assert Path(paths["overlay_polar_hard_path"]).is_file()
    assert paths["overlay_fundus_path"] == paths["overlay_gt_fundus_path"] == ""
    assert _load_json(paths["overlay_metadata_path"])["fundus"] == {}


@pytest.mark.parametrize("mode", ["prediction", "groundtruth"])
@pytest.mark.parametrize("size", ["input", "original"])
def test_original_metadata_must_match_actual_rgb_dimensions_even_for_input_output(tmp_path, mode, size):
    source, _ = _image(tmp_path)
    root = tmp_path / "exports"
    with pytest.raises(ValueError, match="original_size|size|shape|dimension"):
        _save(root, _cfg(mode=mode, size=size), image_path=source, original_size=(320, 240))
    assert not root.exists()


@pytest.mark.parametrize("source_kind", ["none", "missing", "not_an_image"])
def test_enabled_fundus_rejects_missing_or_unreadable_source_before_outputs(tmp_path, source_kind):
    source = None if source_kind == "none" else tmp_path / "source.png"
    if source_kind == "not_an_image":
        source.write_text("not a PNG")
    root = tmp_path / "exports"
    with pytest.raises((ValueError, OSError)):
        _save(root, _cfg(), image_path=source)
    assert not root.exists()


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan"), float("inf")])
def test_enabled_fundus_rejects_invalid_opacity(tmp_path, alpha):
    with pytest.raises(ValueError, match="alpha"):
        QuantificationOverlayWriter(_cfg(alpha=alpha), tmp_path)


@pytest.mark.parametrize("value", ["polar", "cartesian", "prediction_and_gt", 1])
def test_fundus_mode_rejects_other_export_vocabularies(value):
    with pytest.raises(ValueError):
        normalize_fundus_mode(value)


def test_custom_destinations_and_gt_only_need_no_prediction_source(tmp_path):
    source, _ = _image(tmp_path)
    cfg = _cfg(mode="groundtruth")
    readout = _readout(cfg)
    mask = _groundtruth()
    writer = QuantificationOverlayWriter(
        cfg, tmp_path / "exports", fundus_dir=tmp_path / "prediction_custom",
        gt_fundus_dir=tmp_path / "gt_custom",
    )
    paths = writer.save(None, "case", readout=readout, hard=None, soft=None,
                        image_path=source, prediction_source=None,
                        groundtruth_mask=mask, groundtruth_hard=_hard(readout, mask))
    assert Path(paths["overlay_gt_fundus_path"]) == tmp_path / "gt_custom" / "case.png"
    assert paths["overlay_fundus_path"] == ""
    assert not (tmp_path / "prediction_custom").exists()
    record = _load_json(paths["overlay_metadata_path"])["fundus"]
    assert record["prediction_source"] is None
    assert record["prediction"] == {}


def test_fundus_requires_the_requested_hard_data_and_groundtruth(tmp_path):
    source, _ = _image(tmp_path)
    cfg = _cfg()
    readout = _readout(cfg)
    writer = QuantificationOverlayWriter(cfg, tmp_path / "prediction")
    with pytest.raises(ValueError, match="hard"):
        writer.save(_mask(), "case", readout=readout, hard=None, soft=None, image_path=source)
    cfg["eval"]["save_fundus_overlays"] = "groundtruth"
    writer = QuantificationOverlayWriter(cfg, tmp_path / "groundtruth")
    with pytest.raises(ValueError, match="[Gg]round"):
        writer.save(None, "case", readout=readout, hard=None, soft=None, image_path=source)
    assert not (tmp_path / "prediction").exists()
    assert not (tmp_path / "groundtruth").exists()
