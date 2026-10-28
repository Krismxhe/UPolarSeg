"""Both Dv definitions and exact frozen fixed-vertical numerical parity.

The original pre-split implementation lives in tests/fixtures and is never
imported by application code. UPOLARSEG_REFERENCE_ROOT optionally overrides
that reference with an untouched project snapshot. No sibling checkout is
needed to run the complete suite from a delivered code package. Fixed-vertical
measurement remains the default; maximum central chords require an explicit mode.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.losses.polar_quantification import (
    QUANT_VAL_KEYS,
    PolarQuantificationLoss,
    build_polar_quantification,
)
from src.quantification.diagnostics import compute_quantification_diagnostics
from src.quantification.hard_readout import measure_hard_mask
from src.quantification.polar_readout import PolarReadout, build_polar_readout
from src.quantification.soft_readout import read_soft_probabilities


@pytest.fixture(scope="module")
def original_module():
    configured = os.environ.get("UPOLARSEG_REFERENCE_ROOT")
    path = (
        Path(configured) / "src/losses/polar_quantification.py"
        if configured
        else Path(__file__).resolve().parent / "fixtures/polar_quantification_before_split.py"
    )
    if not path.is_file():
        pytest.fail(f"Frozen numerical reference not found: {path}")
    spec = importlib.util.spec_from_file_location("_upolarseg_original_quantification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readout(readout_type=PolarReadout, *, truncated=False, dv_mode=None):
    valid = torch.ones(8, 32, dtype=torch.bool)
    if truncated:
        valid[:, 28:] = False
    kwargs = {"dv_mode": dv_mode} if dv_mode is not None else {}
    return readout_type(valid_grid=valid, max_radius=32.0, vertical_angle_deg=90.0, **kwargs)


def _readable_mask():
    masks = torch.zeros(1, 8, 32, dtype=torch.long)
    for angle, radius in enumerate((15, 9, 6, 8, 13, 10, 12, 7)):
        masks[:, angle, :radius + 1] = 1
    masks[:, 0, 16:21] = 2
    return masks


def _mixed_masks():
    masks = _readable_mask().repeat(6, 1, 1)
    masks[1] = 0  # Empty OD: no valid reference.
    masks[2, 2, 3] = 0  # Broken OD radial interval.
    masks[3, 0, 17] = 0  # Hard reference readable, occupancy incompatible.
    masks[4, 0, 27] = 2  # Foreground reaches the last valid radial sample.
    masks[5, 0, 29] = 2  # Foreground outside the image-valid grid.
    return masks


def _logits(batch_size):
    generator = torch.Generator().manual_seed(20261028)
    return torch.randn(batch_size, 3, 8, 32, generator=generator)


def _assert_tensors_equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _assert_readouts_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        _assert_tensors_equal(actual[name], expected[name])


def _assert_metrics_equal(actual, expected):
    assert tuple(actual) == tuple(expected)
    for name in expected:
        _assert_tensors_equal(actual[name][0], expected[name][0])
        _assert_tensors_equal(actual[name][1], expected[name][1])


@pytest.mark.parametrize("dv_mode,expected_dv", [
    (None, 18.0), ("fixed_vertical", 18.0), ("max_center_chord", 28.0),
])
def test_known_hard_measurement_and_compatible_one_hot_readout(dv_mode, expected_dv):
    readout = _readout(dv_mode=dv_mode)
    masks = _readable_mask()
    hard = measure_hard_mask(readout, masks)
    assert hard["reference_valid"].item()
    assert hard["aux_valid"].item()
    assert hard["pmw"].item() == 5.0
    # The opposite radii 15 and 13 produce the maximum central chord.
    # The old fixed-vertical chord is only 6 + 12 = 18.
    assert readout.dv_mode == (dv_mode or "fixed_vertical")
    assert hard["dv"].item() == expected_dv
    torch.testing.assert_close(hard["pvdr"], torch.tensor([5.0 / expected_dv]))

    probabilities = torch.nn.functional.one_hot(masks, 3).permute(0, 3, 1, 2).float()
    soft = read_soft_probabilities(readout, probabilities)
    for name in ("pmw", "dv", "pvdr", "od_profile", "width_profile"):
        _assert_tensors_equal(soft[name], hard[name])
    _assert_readouts_equal(readout.measure_mask(masks), hard)
    _assert_readouts_equal(readout.read_probabilities(probabilities), soft)
    assert not any(value.requires_grad for value in hard.values())


def test_hard_reference_and_soft_surrogate_keep_distinct_validity_semantics():
    readout = _readout(truncated=True)
    masks = _mixed_masks()
    hard = readout.measure_mask(masks)
    assert hard["reference_valid"].tolist() == [True, False, False, True, False, False]
    assert hard["aux_valid"].tolist() == [True, False, False, False, False, False]
    probabilities = torch.nn.functional.one_hot(masks, 3).permute(0, 3, 1, 2).float()
    soft = readout.read_probabilities(probabilities)
    assert hard["pmw"][3].item() == 5.0
    assert soft["pmw"][3].item() == 4.0


def test_standalone_soft_readout_preserves_gradients():
    readout = _readout()
    logits = _logits(2).requires_grad_()
    soft = read_soft_probabilities(readout, logits.softmax(dim=1))
    objective = sum(soft[name].sum() for name in ("pmw", "dv", "pvdr"))
    objective.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("dv_mode,expected_dv,endpoints", [
    ("max_center_chord", 28.0, (0, 4)),
    ("fixed_vertical", 18.0, (2, 6)),
])
def test_shared_summary_remains_differentiable(dv_mode, expected_dv, endpoints):
    readout = _readout(dv_mode=dv_mode)
    od = torch.tensor([[15.0, 9.0, 6.0, 8.0, 13.0, 10.0, 12.0, 7.0]], requires_grad=True)
    width = torch.tensor([[5.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    result = readout._summarize(od, width)
    assert result["dv"].item() == expected_dv
    sum(result[name].sum() for name in ("pmw", "dv", "pvdr")).backward()
    assert torch.isfinite(od.grad).all()
    assert torch.isfinite(width.grad).all()
    assert all(od.grad[0, index] != 0 for index in endpoints)
    assert all(od.grad[0, index] == 0 for index in range(8) if index not in endpoints)
    assert width.grad[0, 0] != 0
    assert result["od_profile"] is od
    assert result["width_profile"] is width


def test_diagnostics_retain_distinct_metric_populations_and_no_grad():
    readout = _readout(truncated=True)
    masks = _mixed_masks()
    predicted = masks.clone()
    predicted[0, 2, 3] = 0
    logits = torch.nn.functional.one_hot(predicted, 3).permute(0, 3, 1, 2).float()
    logits = (logits * 4.0).requires_grad_()
    result = compute_quantification_diagnostics(readout, logits, masks)
    assert tuple(result) == ("target", "hard", "soft", "gt_soft", "metrics")
    assert tuple(result["metrics"]) == QUANT_VAL_KEYS
    assert result["metrics"]["pmw_soft_mae_px"][1][0].item()
    assert not result["metrics"]["pmw_hard_mae_px"][1][0].item()
    assert result["metrics"]["hard_invalid_fraction"][0][0].item() == 1.0
    for name in ("target", "hard", "soft", "gt_soft"):
        assert not any(value.requires_grad for value in result[name].values())
    for values, valid in result["metrics"].values():
        assert not values.requires_grad and not valid.requires_grad


def test_constructor_buffers_and_state_dict_are_compatible(original_module):
    original = _readout(original_module.PolarQuantificationLoss, truncated=True)
    updated = _readout(PolarQuantificationLoss, truncated=True, dv_mode="fixed_vertical")
    updated_signature = inspect.signature(type(updated))
    assert updated_signature.parameters["dv_mode"].default == "fixed_vertical"
    legacy_signature = updated_signature.replace(parameters=[
        parameter for name, parameter in updated_signature.parameters.items() if name != "dv_mode"
    ])
    assert inspect.signature(type(original)) == legacy_signature
    assert isinstance(updated, PolarReadout)
    names = ("valid_grid", "ray_weights", "radii", "last_valid")
    assert tuple(dict(updated.named_buffers())) == names
    for name in names:
        _assert_tensors_equal(getattr(updated, name), getattr(original, name))
    assert list(updated.parameters()) == []
    assert list(updated.state_dict()) == list(original.state_dict()) == []
    updated.load_state_dict(original.state_dict(), strict=True)
    original.load_state_dict(updated.state_dict(), strict=True)
    updated.to(dtype=torch.float64)
    assert updated.radii.dtype == torch.float64
    assert updated.valid_grid.dtype == torch.bool
    assert updated.last_valid.dtype == torch.int64


def test_hard_soft_and_diagnostics_match_original(original_module):
    original = _readout(original_module.PolarQuantificationLoss, truncated=True)
    updated = _readout(PolarQuantificationLoss, truncated=True, dv_mode="fixed_vertical")
    masks = _mixed_masks()
    logits = _logits(masks.shape[0])
    probabilities = logits.softmax(dim=1)
    _assert_readouts_equal(updated.measure_mask(masks), original.measure_mask(masks))
    _assert_readouts_equal(updated.read_probabilities(probabilities), original.read_probabilities(probabilities))
    expected = original.diagnostics(logits, masks)
    _assert_metrics_equal(updated.diagnostics(logits, masks), expected)
    result = compute_quantification_diagnostics(updated, logits, masks)
    _assert_metrics_equal(result["metrics"], expected)
    _assert_readouts_equal(result["target"], original.measure_mask(masks))
    _assert_readouts_equal(result["hard"], original.measure_mask(logits.argmax(dim=1)))
    _assert_readouts_equal(result["soft"], original.read_probabilities(probabilities))
    gt_probabilities = torch.stack([masks == k for k in range(3)], dim=1).float()
    _assert_readouts_equal(result["gt_soft"], original.read_probabilities(gt_probabilities))


@pytest.mark.parametrize("all_invalid", [False, True])
def test_loss_and_logit_gradients_match_original(original_module, all_invalid):
    original = _readout(original_module.PolarQuantificationLoss, truncated=True)
    updated = _readout(PolarQuantificationLoss, truncated=True, dv_mode="fixed_vertical")
    masks = _mixed_masks()
    if all_invalid:
        masks = masks[[1, 2, 3, 4, 5]]
    old_logits = _logits(masks.shape[0]).requires_grad_()
    new_logits = old_logits.detach().clone().requires_grad_()
    old_loss, old_stats = original(old_logits, masks)
    new_loss, new_stats = updated(new_logits, masks)
    old_loss.backward()
    new_loss.backward()
    _assert_tensors_equal(new_loss, old_loss)
    _assert_readouts_equal(new_stats, old_stats)
    _assert_tensors_equal(new_logits.grad, old_logits.grad)
    assert torch.isfinite(new_logits.grad).all()
    if all_invalid:
        assert new_loss.item() == 0.0
        assert torch.count_nonzero(new_logits.grad).item() == 0
    else:
        assert new_logits.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "k,mask_kind",
    [(1, "all"), (10, "vector"), (None, "per_case"), (3, "none"), (100, "vector")],
)
def test_candidate_top_k_masks_and_gradients_match_original(original_module, k, mask_kind):
    original = _readout(original_module.PolarQuantificationLoss, truncated=True)
    updated = _readout(PolarQuantificationLoss, truncated=True, dv_mode="fixed_vertical")
    masks = _mixed_masks()
    old_logits = _logits(masks.shape[0]).requires_grad_()
    new_logits = old_logits.detach().clone().requires_grad_()
    if mask_kind == "all":
        angle_mask = None
    elif mask_kind == "vector":
        angle_mask = torch.tensor([True, False, True, False, False, False, False, False])
    elif mask_kind == "none":
        angle_mask = torch.zeros(masks.shape[0], 8, dtype=torch.bool)
    else:
        angle_mask = torch.ones(masks.shape[0], 8, dtype=torch.bool)
        angle_mask[0] = False
        angle_mask[1, 3:] = False
    old_error, old_valid = original._candidate_error(
        original.read_probabilities(old_logits.softmax(dim=1)),
        original.measure_mask(masks), k=k, angle_mask=angle_mask,
    )
    new_error, new_valid = updated._candidate_error(
        updated.read_probabilities(new_logits.softmax(dim=1)),
        updated.measure_mask(masks), k=k, angle_mask=angle_mask,
    )
    old_error.sum().backward()
    new_error.sum().backward()
    _assert_tensors_equal(new_error, old_error)
    _assert_tensors_equal(new_valid, old_valid)
    _assert_tensors_equal(new_logits.grad, old_logits.grad)


def _geometry_config():
    return {
        "dataset": {"num_classes": 3, "mask_mode": "index"},
        "train": {"img_size": 32},
        "polar": {
            "enabled": True,
            "geometry": {
                "mode": "linear", "angle_bins": 8, "radius_bins": 32,
                "center": [11.25, 13.5], "max_radius": 10.0,
            },
        },
        "quantification": {"enabled": False, "weight": float("nan")},
    }


def _require_geometry_dependencies():
    pytest.importorskip("cv2")
    pytest.importorskip("albumentations")


def test_independent_builder_ignores_loss_enable_flag_and_weight():
    _require_geometry_dependencies()
    cfg = _geometry_config()
    readout = build_polar_readout(cfg)
    assert type(readout) is PolarReadout
    assert readout.geometry_metadata["center"] == (11.25, 13.5)
    assert readout.geometry_metadata["img_size"] == 32
    assert readout.geometry_metadata["mode"] == "linear"
    assert readout.dv_mode == "fixed_vertical"
    assert not readout.state_dict()
    assert build_polar_quantification(cfg) is None
    cfg["quantification"]["enabled"] = True
    with pytest.raises(ValueError, match="weight"):
        build_polar_quantification(cfg)


def test_loss_builder_disabled_path_stays_lazy():
    assert build_polar_quantification({}) is None
    assert build_polar_quantification({"quantification": {"enabled": False, "weight": "invalid"}}) is None


@pytest.mark.parametrize("dv_mode,expected_mode,expected_dv", [
    (None, "fixed_vertical", 18.0),
    ("fixed_vertical", "fixed_vertical", 18.0),
    ("max_center_chord", "max_center_chord", 28.0),
])
def test_loss_and_readout_builders_use_the_same_configured_dv(dv_mode, expected_mode, expected_dv):
    _require_geometry_dependencies()
    cfg = _geometry_config()
    cfg["quantification"] = {"enabled": True, "weight": 0.1}
    if dv_mode is not None:
        cfg["quantification"]["dv_mode"] = dv_mode
    loss = build_polar_quantification(copy.deepcopy(cfg))
    readout = build_polar_readout(copy.deepcopy(cfg))
    assert loss.dv_mode == readout.dv_mode == expected_mode
    masks = _readable_mask()
    actual = readout.measure_mask(masks)
    _assert_readouts_equal(loss.measure_mask(masks), actual)
    assert actual["dv"].item() == expected_dv * readout.max_radius / readout.radius_bins


@pytest.mark.parametrize("bad_config", ["classes", "mask_mode", "polar_disabled", "log_grid"])
def test_independent_builder_keeps_geometry_preconditions(bad_config):
    cfg = _geometry_config()
    if bad_config == "classes":
        cfg["dataset"]["num_classes"] = 2
    elif bad_config == "mask_mode":
        cfg["dataset"]["mask_mode"] = "binary"
    elif bad_config == "polar_disabled":
        cfg["polar"]["enabled"] = False
    else:
        cfg["polar"]["geometry"]["mode"] = "log"
    with pytest.raises(ValueError):
        build_polar_readout(cfg)


@pytest.mark.parametrize("explicit_center", [True, False])
def test_both_builders_match_original_geometry(original_module, explicit_center):
    _require_geometry_dependencies()
    cfg = _geometry_config()
    cfg["quantification"] = {"enabled": True, "weight": 0.1, "vertical_angle_deg": 112.5}
    if not explicit_center:
        cfg["polar"]["geometry"].pop("center")
        cfg["polar"]["geometry"].pop("max_radius")
    original = original_module.build_polar_quantification(copy.deepcopy(cfg))
    cfg["quantification"]["dv_mode"] = "fixed_vertical"
    loss = build_polar_quantification(copy.deepcopy(cfg))
    readout = build_polar_readout(copy.deepcopy(cfg))
    for updated in (loss, readout):
        assert updated.dv_mode == "fixed_vertical"
        for name, value in original.named_buffers():
            _assert_tensors_equal(getattr(updated, name), value)
        for name in ("max_radius", "angle_bins", "radius_bins", "vertical_angle_deg", "denominator_floor_px"):
            assert getattr(updated, name) == getattr(original, name)
    if not explicit_center:
        assert readout.geometry_metadata["center"] == (15.5, 15.5)
