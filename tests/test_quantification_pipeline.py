"""Contracts across model outputs, polar diagnostics and per-case collection.

The readout/record tests only require torch. SegModule integration tests use the
real Lightning module with tiny injected networks, without a Trainer, downloads
or a training dataset. Native OD/PPA records use small temporary PNG masks.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from src.evaluation.quantification_records import (
    QUANTIFICATION_FIELDS,
    QUANTIFICATION_METRIC_SPECS,
    build_quantification_case_records,
    quantification_protocol,
    unavailable_quantification_record,
)
from src.quantification.diagnostics import compute_quantification_diagnostics
from src.quantification.polar_readout import PolarReadout


def _base_mask():
    """A centred OD with fixed-axis Dv=20, max-chord Dv=21, and PMW=5."""
    mask = torch.zeros(16, 32, dtype=torch.long)
    for angle in range(16):
        radius = 5 + angle % 7
        width = 2 + angle % 4
        mask[angle, : radius + 1] = 1
        mask[angle, radius + 1 : radius + width + 1] = 2
    return mask


def _logits(masks):
    return F.one_hot(masks, num_classes=3).permute(0, 3, 1, 2).float() * 16.0


def _readout(*, dv_mode=None, vertical_angle_deg=101.25):
    grid = torch.ones(16, 32, dtype=torch.bool)
    grid[:, 28:] = False
    kwargs = {"dv_mode": dv_mode} if dv_mode is not None else {}
    return PolarReadout(grid, max_radius=32.0, vertical_angle_deg=vertical_angle_deg, **kwargs)


def _rows(readout, logits, masks, source="seg_logits"):
    return build_quantification_case_records(
        compute_quantification_diagnostics(readout, logits, masks),
        protocol=quantification_protocol(readout),
        prediction_source=source,
    )


def _assert_scalar_record(record):
    for value in record.values():
        if isinstance(value, dict):
            _assert_scalar_record(value)
        else:
            assert not isinstance(value, torch.Tensor)
            assert isinstance(value, (str, bool, int, float)), type(value)


@pytest.mark.parametrize("dv_mode,expected_dv,expected_protocol", [
    (None, 20.0, "polar-radial-v1"),
    ("max_center_chord", 21.0, "polar-radial-v2"),
])
def test_case_records_preserve_measurements_and_invalidity_separately(dv_mode, expected_dv, expected_protocol):
    readout = _readout(dv_mode=dv_mode)
    masks = _base_mask().unsqueeze(0).repeat(6, 1, 1)
    masks[1][masks[1] == 2] = 0
    masks[2, 0, 2] = 0  # disconnected OD
    masks[3, 0, 6] = 0  # readable outer PPA boundary, incompatible soft target
    masks[4, 1, 27] = 2  # clipped at the last image-valid radial sample
    masks[5, 1, 30] = 2  # foreground outside the image-valid grid
    rows = _rows(readout, _logits(masks), masks)

    assert len(rows) == 6
    assert [r["quant_reference_valid"] for r in rows] == [True, True, False, True, False, False]
    assert [r["quant_integral_compatible"] for r in rows] == [True, True, False, False, False, True]
    assert [r["quant_aux_valid"] for r in rows] == [True, True, False, False, False, False]
    assert rows[0]["quant_reference_pmw"] == 5.0
    assert rows[0]["quant_reference_dv"] == expected_dv
    assert rows[0]["quant_reference_pvdr"] == pytest.approx(5.0 / expected_dv)
    assert rows[1]["quant_reference_pmw"] == 0.0  # a valid PPA-negative measurement
    assert rows[1]["quant_pmw_hard_mae_px"] == 0.0
    assert rows[1]["quant_pmw_hard_mae_px_valid"] is True
    assert rows[3]["quant_reference_pmw"] == 5.0
    assert rows[3]["quant_pmw_hard_mae_px_valid"] is True
    for index in (2, 4, 5):
        assert math.isnan(rows[index]["quant_reference_pmw"])
        assert math.isnan(rows[index]["quant_hard_pmw"])
        assert math.isnan(rows[index]["quant_pmw_hard_mae_px"])
        assert rows[index]["quant_pmw_hard_mae_px_valid"] is False
        assert math.isfinite(rows[index]["quant_soft_pmw"])
    for row in rows:
        assert set(QUANTIFICATION_FIELDS) <= set(row)
        assert row["quant_prediction_source"] == "seg_logits"
        assert row["quant_status"] == "ok"
        assert row["quant_protocol"] == expected_protocol
        _assert_scalar_record(row)


def test_paired_soft_error_uses_the_same_valid_cases_as_hard_error():
    masks = _base_mask().unsqueeze(0).repeat(2, 1, 1)
    predictions = masks.clone()
    predictions[1] = 0
    rows = _rows(_readout(), _logits(predictions), masks)

    assert rows[0]["quant_pmw_soft_mae_paired_px_valid"] is True
    assert rows[0]["quant_pmw_soft_mae_paired_px"] == rows[0]["quant_pmw_soft_mae_px"]
    failed = rows[1]
    assert failed["quant_reference_valid"] is True
    assert failed["quant_hard_reference_valid"] is False
    assert failed["quant_pmw_soft_mae_px_valid"] is True
    assert math.isfinite(failed["quant_pmw_soft_mae_px"])
    assert failed["quant_pmw_soft_mae_paired_px_valid"] is False
    assert math.isnan(failed["quant_pmw_soft_mae_paired_px"])
    assert math.isnan(failed["quant_pmw_hard_mae_px"])
    assert failed["quant_hard_invalid_fraction"] == 1.0
    assert failed["quant_hard_invalid_fraction_valid"] is True


def test_unavailable_measurements_are_not_reported_as_zero_error():
    row = unavailable_quantification_record("nonlinear polar geometry")
    assert set(row) == set(QUANTIFICATION_FIELDS)
    assert row["quant_status"] == "unavailable"
    assert row["quant_reason"] == "nonlinear polar geometry"
    for source in ("reference", "hard", "soft"):
        for name in ("pmw", "dv", "pvdr"):
            assert math.isnan(row[f"quant_{source}_{name}"])
    for spec in QUANTIFICATION_METRIC_SPECS:
        assert math.isnan(row[spec["key"]])
        assert row[spec["valid_key"]] is False


def test_protocol_tracks_geometry_and_keeps_polar_pixel_units_explicit():
    first = _readout()
    same_geometry = _readout()
    changed_geometry = _readout(vertical_angle_deg=90.0)
    protocol = quantification_protocol(first)
    assert protocol == quantification_protocol(same_geometry)
    assert protocol["quant_geometry_id"] != quantification_protocol(changed_geometry)["quant_geometry_id"]
    assert protocol["quant_protocol"] == "polar-radial-v1"
    changed_geometry = _readout()
    changed_geometry.valid_grid[0, 27] = False
    assert protocol["quant_geometry_id"] != quantification_protocol(changed_geometry)["quant_geometry_id"]
    assert protocol["quant_space"] == "resized_roi_polar"
    assert protocol["quant_length_unit"] == "resized_roi_px"
    geometry = json.loads(protocol["quant_geometry_json"])
    assert geometry["radial_step"] == "max_radius/radius_bins"
    assert geometry["vertical_angle_deg"] == 101.25


def test_max_chord_protocol_ignores_legacy_axis_and_separates_measurement_modes():
    readout = _readout(dv_mode="max_center_chord")
    protocol = quantification_protocol(readout)
    changed_axis = _readout(dv_mode="max_center_chord", vertical_angle_deg=90.0)
    assert protocol == quantification_protocol(changed_axis)
    assert protocol["quant_protocol"] == "polar-radial-v2"
    assert protocol["quant_geometry_id"] != quantification_protocol(_readout())["quant_geometry_id"]
    geometry = json.loads(protocol["quant_geometry_json"])
    assert geometry["dv_mode"] == "max_center_chord"
    assert "vertical_angle_deg" not in geometry
    readout.geometry_metadata["vertical_angle_deg"] = 37.0
    assert quantification_protocol(readout) == protocol


def test_records_reject_an_ambiguous_prediction_source():
    masks = _base_mask().unsqueeze(0)
    with pytest.raises(ValueError, match="prediction source"):
        _rows(_readout(), _logits(masks), masks, source="prediction")


class _SegmentationLoss(torch.nn.Module):
    def forward(self, logits, masks):
        value = F.cross_entropy(logits, masks)
        return value, value, value


class _FixedModel(torch.nn.Module):
    def __init__(self, logits, *, tensor_output=False):
        super().__init__()
        self.register_buffer("logits", logits)
        self.register_buffer("feature", torch.ones(logits.shape[0], 4, *logits.shape[-2:]))
        self.tensor_output = tensor_output
        self.calls = []

    def forward(self, images, return_features=False):
        self.calls.append(bool(return_features))
        assert images.shape[0] == self.logits.shape[0]
        if self.tensor_output:
            return self.logits
        return {"seg_logits": self.logits, "features": {"decoder_final": self.feature}}


class _FixedAdapter(torch.nn.Module):
    def __init__(self, model, quant_logits):
        super().__init__()
        self.expected_feature = model.feature
        self.expected_logits = model.logits
        self.register_buffer("quant_logits", quant_logits)
        self.calls = 0

    def forward(self, features, logits):
        assert features is self.expected_feature
        assert logits is self.expected_logits
        self.calls += 1
        return self.quant_logits


class _TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 3, 1)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.projection(images)


class _CaptureMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def update(self, predictions, target):
        self.calls.append((predictions.detach().clone(), target.detach().clone()))


@pytest.fixture
def module_factory(monkeypatch):
    """Construct the real SegModule, replacing only network and loss factories."""
    module = pytest.importorskip("src.models.seg_module")
    omegaconf = pytest.importorskip("omegaconf")

    def make(model, *, quantification=False, adapter=False, odppa=True, include_quant_config=True,
             polar=True, polar_mode="linear", dv_mode=None):
        cfg = omegaconf.OmegaConf.create({
            "dataset": {
                "name": "synthetic_polar", "num_classes": 3, "mask_mode": "index",
                "class_names": ["background", "OD", "PPA"], "foreground_classes": [1, 2],
            },
            "model": {"arch": "test", "decoder_channels": [4]},
            "train": {"img_size": 32},
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "polar": {
                "enabled": polar,
                "geometry": {"mode": polar_mode, "center": None, "max_radius": 32.0,
                             "angle_bins": 16, "radius_bins": 32, "pt_type": "corner"},
                "eval": {"metric_space": "cart"},
            },
            "quantification": {
                "enabled": quantification, "adapter_enabled": adapter, "weight": 0.2,
                "vertical_angle_deg": 101.25, "denominator_floor_px": 1.0,
            },
            "eval": {"odppa": {"enabled": odppa, "metadata_csv": None}},
        })
        if dv_mode is not None:
            cfg.quantification.dv_mode = dv_mode
        if adapter is None:
            del cfg.quantification["adapter_enabled"]
        if not include_quant_config:
            del cfg["quantification"]
        monkeypatch.setattr(module, "build_model", lambda *_: model)
        monkeypatch.setattr(module, "build_loss", lambda *_: _SegmentationLoss())
        instance = module.SegModule(cfg)
        monkeypatch.setattr(instance, "log", lambda *args, **kwargs: None)
        instance.test_dice = _CaptureMetric()
        instance.test_iou = _CaptureMetric()
        return instance

    return make


def _batch(masks, *, tmp_path=None, cfg=None, case_ids=None, patient_ids=None):
    count = masks.shape[0]
    case_ids = case_ids or [f"case_{i}" for i in range(count)]
    mask_paths = ["" for _ in range(count)]
    if tmp_path is not None:
        from PIL import Image
        from src.metrics.odppa_eval import restore_prediction_to_native

        for i, case_id in enumerate(case_ids):
            native = restore_prediction_to_native(masks[i].numpy(), (32, 32), cfg)
            path = tmp_path / f"{case_id}.png"
            Image.fromarray(native.astype("uint8")).save(path)
            mask_paths[i] = str(path)
    batch = {
        "image": torch.zeros(count, 3, *masks.shape[-2:]), "mask": masks,
        "case_id": case_ids, "image_path": [f"/{cid}.png" for cid in case_ids],
        "mask_path": mask_paths,
    }
    if patient_ids is not None:
        batch["patient_id"] = patient_ids
    return batch


@pytest.mark.parametrize("include_quant_config", [True, False])
def test_baseline_builds_a_readout_without_a_quantification_training_object(module_factory, include_quant_config):
    masks = _base_mask().unsqueeze(0)
    model = _FixedModel(_logits(masks), tensor_output=True)
    module = module_factory(model, include_quant_config=include_quant_config)
    assert module.quantification is None
    readout = module._quantification_readout()
    assert isinstance(readout, PolarReadout)
    assert tuple(readout.valid_grid.shape) == (16, 32)
    assert readout.max_radius == 32.0
    assert readout.geometry_metadata["img_size"] == 32
    assert readout.dv_mode == "fixed_vertical"
    assert module.quant_test_metrics
    assert not any(key.startswith("_quant_eval_readout.") for key in module.state_dict())


@pytest.mark.parametrize("dv_mode,expected_dv", [(None, 20.0), ("max_center_chord", 21.0)])
def test_baseline_records_share_case_and_patient_identity_with_odppa(module_factory, tmp_path, dv_mode, expected_dv):
    masks = _base_mask().unsqueeze(0).repeat(3, 1, 1)
    model = _FixedModel(_logits(masks), tensor_output=True)
    module = module_factory(model, dv_mode=dv_mode)
    module._odppa_metadata = {"001": {"patient_id": "007", "eye": "L"}}
    batch = _batch(masks, tmp_path=tmp_path, cfg=module.cfg,
                   case_ids=["001", "002", "003"], patient_ids=["wrong", "008", ""])
    module.test_step(batch, 0)

    assert module.quantification is None
    assert model.calls == [False]
    assert [r["case_id"] for r in module.test_records] == ["001", "002", "003"]
    assert [r["patient_id"] for r in module.test_records] == ["007", "008", "003"]
    for row in module.test_records:
        assert "od_dsc" in row and "ppa_status" in row
        assert row["quant_prediction_source"] == "seg_logits"
        assert row["quant_status"] == "ok"
        assert row["quant_hard_pmw"] == 5.0
        assert row["quant_reference_dv"] == expected_dv
        assert row["quant_pmw_hard_mae_px"] == 0.0
        assert row["quant_pmw_hard_mae_px_valid"] is True
        _assert_scalar_record(row)
    torch.testing.assert_close(module.test_dice.calls[0][0], masks)
    assert module.quant_test_metrics["pmw_hard_mae_px"].compute().item() == 0.0


def test_adapter_test_uses_quant_logits_without_changing_segmentation_predictions(module_factory):
    masks = _base_mask().unsqueeze(0)
    corrected = masks.clone()
    for angle in range(16):
        end = 5 + angle % 7 + 2 + angle % 4
        corrected[0, angle, end + 1 : end + 3] = 2
    model = _FixedModel(_logits(masks))
    module = module_factory(model, quantification=True, adapter=True, odppa=False)
    adapter = _FixedAdapter(model, _logits(corrected))
    module.quant_head = adapter
    module.test_step(_batch(masks, case_ids=["adapter_case"], patient_ids=["patient_1"]), 0)

    assert model.calls == [True]
    assert adapter.calls == 1
    torch.testing.assert_close(module.test_dice.calls[0][0], masks)
    torch.testing.assert_close(module.test_iou.calls[0][0], masks)
    row = module.test_records[0]
    assert row["per_class"][2]["dice"] == 1.0
    assert row["quant_prediction_source"] == "quant_logits"
    assert row["quant_reference_pmw"] == 5.0
    assert row["quant_hard_pmw"] == 7.0
    assert row["quant_pmw_hard_mae_px"] == 2.0
    assert row["patient_id"] == "patient_1"
    assert module.quant_test_metrics["pmw_hard_mae_px"].compute().item() == 2.0


def test_adapter_without_decoder_features_fails_instead_of_silently_using_seg_logits(module_factory):
    masks = _base_mask().unsqueeze(0)
    model = _FixedModel(_logits(masks), tensor_output=True)
    module = module_factory(model, quantification=True, adapter=True, odppa=False)
    with pytest.raises((ValueError, KeyError, RuntimeError), match="feature|decoder"):
        module.test_step(_batch(masks), 0)
    assert not module.test_records


@pytest.mark.parametrize("polar,mode", [(False, "linear"), (True, "log")])
def test_unsupported_baseline_geometry_emits_unavailable_records(module_factory, tmp_path, polar, mode):
    masks = _base_mask().unsqueeze(0)
    model = _FixedModel(_logits(masks))
    module = module_factory(model, polar=polar, polar_mode=mode)
    module.test_step(_batch(masks, tmp_path=tmp_path, cfg=module.cfg), 0)
    row = module.test_records[0]
    assert row["quant_status"] == "unavailable"
    assert row["quant_reason"]
    assert math.isnan(row["quant_pmw_hard_mae_px"])
    assert row["quant_pmw_hard_mae_px_valid"] is False
    assert module.quantification is None
    assert module._quantification_readout() is None


def test_incomplete_ppa_annotation_keeps_dv_and_prediction_readouts(module_factory, tmp_path):
    masks = _base_mask().unsqueeze(0).repeat(4, 1, 1)
    predictions = masks.clone()
    for index in (0, 2):
        for angle in range(16):
            end = 5 + angle % 7 + 2 + angle % 4
            predictions[index, angle, end + 1 : end + 3] = 2
    model = _FixedModel(_logits(predictions))
    module = module_factory(model)
    module._odppa_metadata = {
        "csv_unknown": {"reference_complete": "False"},
        "csv_complete": {"reference_complete": "True"},
    }
    batch = _batch(masks, tmp_path=tmp_path, cfg=module.cfg,
                   case_ids=["csv_unknown", "csv_complete", "batch_unknown", "batch_complete"])
    # A normal torch DataLoader collates scalar bool metadata into a bool tensor.
    batch["reference_complete"] = torch.tensor([True, False, False, True])
    module.test_step(batch, 0)

    expected_complete = [False, True, False, True]
    assert [r["quant_annotation_complete"] for r in module.test_records] == expected_complete
    assert [r["ppa_status"] for r in module.test_records] == ["unknown", "positive", "unknown", "positive"]
    for row, complete in zip(module.test_records, expected_complete):
        assert row["quant_reference_valid"] is True
        assert row["quant_reference_dv_valid"] is True
        assert row["quant_reference_dv"] == 20.0
        assert row["quant_dv_hard_mae_px"] == 0.0
        assert row["quant_dv_hard_mae_px_valid"] is True
        assert row["quant_hard_reference_valid"] is True
        assert row["quant_hard_pmw"] == (5.0 if complete else 7.0)
        assert math.isfinite(row["quant_soft_pmw"])
        for name, target in (("pmw", "PMW"), ("pvdr", "PVDR")):
            assert row[f"quant_reference_{name}_valid"] is complete
            if not complete:
                assert math.isnan(row[f"quant_reference_{name}"])
                for spec in QUANTIFICATION_METRIC_SPECS:
                    if spec["target"] == target:
                        assert math.isnan(row[spec["key"]])
                        assert row[spec["valid_key"]] is False
    # Only the two fully annotated cases belong to PMW MAE, in both CSV and logs.
    assert module.quant_test_metrics["pmw_hard_mae_px"].compute().item() == 0.0


def test_all_unknown_ppa_labels_leave_pmw_logging_undefined_not_zero(module_factory, tmp_path):
    masks = _base_mask().unsqueeze(0)
    module = module_factory(_FixedModel(_logits(masks)))
    batch = _batch(masks, tmp_path=tmp_path, cfg=module.cfg)
    batch["reference_complete"] = [False]
    module.test_step(batch, 0)
    assert math.isnan(module.test_records[0]["quant_pmw_hard_mae_px"])
    assert torch.isnan(module.quant_test_metrics["pmw_hard_mae_px"].compute()).item()
    assert module.quant_test_metrics["dv_hard_mae_px"].compute().item() == 0.0


def test_non_adapter_quantification_training_calls_the_model_and_backpropagates(module_factory):
    torch.manual_seed(7)
    model = _TrainModel()
    module = module_factory(model, quantification=True, adapter=False, odppa=False)
    masks = _base_mask().unsqueeze(0).repeat(2, 1, 1)
    images = torch.randn(2, 3, 16, 32)
    value = module.training_step((images, masks), 0)

    assert model.calls == 1  # catches the original normalize_model_output(images) branch
    assert value.requires_grad and torch.isfinite(value)
    value.backward()
    gradient = model.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("quantification,adapter", [(False, None), (False, True), (True, None)])
def test_readout_or_disabled_adapter_adds_no_optimizer_parameters(module_factory, quantification, adapter):
    model = _TrainModel()
    module = module_factory(model, quantification=quantification, adapter=adapter)
    optimizer = module.configure_optimizers()
    assert len(optimizer.param_groups) == 1
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}
    assert not hasattr(module, "quant_head")


@pytest.mark.parametrize("rank", [0, 1])
def test_ddp_collection_deduplicates_cases_without_losing_quantification_or_patient_group(module_factory, monkeypatch, rank):
    masks = _base_mask().unsqueeze(0).repeat(2, 1, 1)
    module = module_factory(_FixedModel(_logits(masks)), quantification=True, odppa=False)
    module.test_step(_batch(masks, case_ids=["001", "002"], patient_ids=["007", "007"]), 0)
    first, second = module.test_records
    module.test_records = [first]
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)

    def gather(destination, local):
        assert local == [first]
        destination[:] = [[first], [dict(first), second]]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    module._gather_test_records()
    if rank:
        assert module.test_records == []
    else:
        assert [r["case_id"] for r in module.test_records] == ["001", "002"]
        assert [r["patient_id"] for r in module.test_records] == ["007", "007"]
        for row in module.test_records:
            assert row["quant_prediction_source"] == "seg_logits"
            assert row["quant_pmw_hard_mae_px"] == 0.0
            assert row["quant_pmw_hard_mae_px_valid"] is True

