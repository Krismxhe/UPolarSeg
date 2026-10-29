"""Real SegModule and export-entry checks using tiny deterministic networks.

Only the network/checkpoint, loader and Trainer boundaries are substituted;
batch selection, readouts, rendering, metadata and CSV serialization are real.
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")


def _mask(extra_width=0):
    result = torch.zeros(16, 32, dtype=torch.long)
    result[:, :7] = 1
    result[:, 7:10 + extra_width] = 2
    return result


def _logits(labels):
    return torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 16


class _Network(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.register_buffer("logits", _logits(_mask().unsqueeze(0).repeat(count, 1, 1)))
        self.register_buffer("feature", torch.ones(count, 4, 16, 32))
        self.calls = []

    def forward(self, images, return_features=False):
        self.calls.append(bool(return_features))
        return {"seg_logits": self.logits, "features": {"decoder_final": self.feature}}


class _Adapter(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.register_buffer("logits", _logits(_mask(2).unsqueeze(0).repeat(count, 1, 1)))
        self.calls = 0

    def forward(self, features, logits):
        self.calls += 1
        return self.logits


class _Loss(torch.nn.Module):
    def forward(self, logits, labels):
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss, loss, loss


@pytest.fixture
def module_factory(monkeypatch, tmp_path):
    implementation = importlib.import_module("src.models.seg_module")

    def make(*, fundus="none", source="seg_logits", adapter=False, count=1,
             polar=True, cfg=None, old_overlay="none", old_readouts="soft"):
        if cfg is None:
            cfg = omegaconf.OmegaConf.create({
                "dataset": {
                    "name": "tiny_polar", "num_classes": 3, "mask_mode": "index",
                    "class_names": ["background", "OD", "PPA"], "foreground_classes": [1, 2],
                },
                "model": {"arch": "tiny", "decoder_channels": [4]},
                "train": {"img_size": 64},
                "polar": {"enabled": polar, "geometry": {
                    "center": [27.25, 31.5], "max_radius": 32.0,
                    "radius_bins": 32, "angle_bins": 16, "pt_type": "corner", "mode": "linear",
                }, "eval": {"metric_space": "cart"}},
                "quantification": {
                    "enabled": adapter, "adapter_enabled": adapter, "weight": 0.1,
                    "dv_mode": "max_center_chord", "denominator_floor_px": 1.0,
                },
                "eval": {
                    "save_csv": False, "save_predictions": "none",
                    "save_fundus_overlays": fundus, "fundus_overlay_alpha": 0.0,
                    "save_quantification_overlays": old_overlay,
                    "save_quantification_gt_overlays": "none",
                    "quantification_overlay_layout": "separate",
                    "quantification_overlay_readouts": old_readouts,
                    "quantification_overlay_source": source,
                    "quantification_overlay_legend": False,
                    "prediction_cartesian_size": "original",
                    "output_dir": str(tmp_path / "exports"),
                    "odppa": {"enabled": False, "bootstrap_samples": 8, "permutation_samples": 8},
                },
                "logging": {"save_dir": str(tmp_path), "name": "tiny", "logger": "tensorboard"},
                "hardware": {"accelerator": "cpu", "devices": 1, "num_nodes": 1, "strategy": "auto"},
                "split": "test", "checkpoint": "unused.ckpt",
            })
        network = _Network(count)
        monkeypatch.setattr(implementation, "build_model", lambda *args: network)
        monkeypatch.setattr(implementation, "build_loss", lambda *args: _Loss())
        model = implementation.SegModule(cfg).eval()
        monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
        if bool(cfg.get("quantification", {}).get("adapter_enabled", False)):
            model.quant_head = _Adapter(count)
        return model

    return make


def _batch(tmp_path, count=1, *, complete=True, orig_size=True, target_width=1):
    samples = []
    sizes = [(48, 80), (64, 40)]
    for index in range(count):
        shape = sizes[index % 2]
        image_path = tmp_path / f"rgb_{index}.png"
        Image.new("RGB", (shape[1], shape[0]), (31 + index, 68, 112)).save(image_path)
        sample = {
            "image": torch.zeros(3, 16, 32), "mask": _mask(target_width),
            "case_id": f"{index + 1:03d}", "image_path": str(image_path), "mask_path": "",
            "reference_complete": str(complete).lower(),
        }
        if orig_size:
            sample["orig_size"] = shape
        samples.append(sample)
    return torch.utils.data.default_collate(samples)


def _calls(monkeypatch, readout):
    counts = {"hard": 0, "soft": 0}
    for label, name in (("hard", "measure_mask"), ("soft", "read_probabilities")):
        original = getattr(readout, name)
        def counted(*args, _label=label, _original=original, **kwargs):
            counts[_label] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(readout, name, counted)
    return counts


def _metadata(row):
    return json.loads(Path(row["overlay_metadata_path"]).read_text())


@pytest.mark.parametrize("mode,hard_reads", [("prediction", 1), ("groundtruth", 1), ("both", 2)])
def test_fundus_only_routes_each_rgb_and_size_without_soft_or_mask_exports(
    module_factory, monkeypatch, tmp_path, mode, hard_reads,
):
    model = module_factory(fundus=mode, count=2)
    assert not model.quant_test_metrics
    assert model._odppa_eval_enabled is False
    counts = _calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    model.test_step(_batch(tmp_path, count=2), 0)
    assert counts == {"hard": hard_reads, "soft": 0}
    assert model.model.calls == [False]
    assert model._prediction_writer is None
    assert len(model.test_records) == 2
    for index, (row, shape) in enumerate(zip(model.test_records, ((48, 80), (64, 40)), strict=True)):
        assert not any(key.startswith("quant_") for key in row)
        assert row["pred_path"] == row["pred_polar_path"] == row["pred_cartesian_path"] == ""
        assert row["overlay_polar_path"] == row["overlay_cartesian_path"] == ""
        metadata = _metadata(row)
        assert metadata["readouts"] == metadata["groundtruth"] == {}
        fundus = metadata["fundus"]
        assert Path(fundus["source_image"]) == tmp_path / f"rgb_{index}.png"
        assert fundus["original_shape"] == fundus["output_shape"] == list(shape)
        for key, selected in (
            ("overlay_fundus_path", mode in {"prediction", "both"}),
            ("overlay_gt_fundus_path", mode in {"groundtruth", "both"}),
        ):
            assert bool(row[key]) is selected
            if selected:
                with Image.open(row[key]) as image:
                    assert image.mode == "RGB"
                    assert (image.height, image.width) == shape
        if mode in {"prediction", "both"}:
            assert fundus["prediction"]["hard"]["raw"]["pmw"] == 3.0
        if mode in {"groundtruth", "both"}:
            assert fundus["groundtruth"]["hard"]["raw"]["pmw"] == 4.0
        assert all(not isinstance(value, (np.ndarray, torch.Tensor)) for value in row.values())


@pytest.mark.parametrize("adapter,source,actual_source,pmw,expected", [
    (False, "quantification", "seg_logits", 3.0, {"hard": 1, "soft": 0}),
    (True, "seg_logits", "seg_logits", 3.0, {"hard": 3, "soft": 2}),
    (True, "quantification", "quant_logits", 5.0, {"hard": 2, "soft": 2}),
])
def test_fundus_only_prediction_source_matches_masks_and_reuses_diagnostics(
    module_factory, monkeypatch, tmp_path, adapter, source, actual_source, pmw, expected,
):
    model = module_factory(fundus="prediction", adapter=adapter, source=source)
    calls = _calls(monkeypatch, model._quantification_readout())
    model.on_test_epoch_start()
    saved = []
    writer = model._quantification_overlay_writer
    original = writer.save
    def capture(mask, case_id, **kwargs):
        saved.append((mask.copy(), kwargs))
        return original(mask, case_id, **kwargs)
    monkeypatch.setattr(writer, "save", capture)
    model.test_step(_batch(tmp_path), 0)
    assert calls == expected
    assert model.model.calls == [adapter]
    if adapter:
        assert model.quant_head.calls == 1
    np.testing.assert_array_equal(saved[0][0], _mask(2 if actual_source == "quant_logits" else 0))
    assert saved[0][1]["prediction_source"] == actual_source
    assert Path(saved[0][1]["image_path"]) == tmp_path / "rgb_0.png"
    assert float(saved[0][1]["hard"]["pmw"]) == pmw
    # The old readouts='soft' setting has no effect when no old panel is on.
    assert saved[0][1]["soft"] is None
    fundus = _metadata(model.test_records[0])["fundus"]
    assert fundus["prediction_source"] == actual_source
    assert fundus["prediction"]["hard"]["raw"]["pmw"] == pmw


def test_gt_only_batch_requires_no_prediction_logits(module_factory, monkeypatch, tmp_path):
    model = module_factory(fundus="groundtruth")
    calls = _calls(monkeypatch, model._quantification_readout())
    payload = model._quantification_overlay_batch(None, None, None, target_masks=_mask(1).unsqueeze(0))
    assert calls == {"hard": 1, "soft": 0}
    assert payload["mask"] is payload["hard"] is payload["soft"] is None
    assert float(payload["groundtruth_hard"]["pmw"][0]) == 4.0
    with pytest.raises(ValueError, match="target masks"):
        model._quantification_overlay_batch(None, None, None)
    assert calls == {"hard": 1, "soft": 0}


@pytest.mark.parametrize("metadata_source", ["batch", "csv"])
def test_partial_gt_status_reaches_fundus_without_disabling_od_lines(
    module_factory, tmp_path, metadata_source,
):
    model = module_factory(fundus="groundtruth")
    if metadata_source == "csv":
        metadata_path = tmp_path / "reference.csv"
        metadata_path.write_text("case_id,reference_complete\n001,false\n")
        model.cfg.eval.odppa.metadata_csv = str(metadata_path)
        model = module_factory(cfg=model.cfg)
    model.on_test_epoch_start()
    model.test_step(_batch(tmp_path, complete=metadata_source == "csv"), 0)
    gt = _metadata(model.test_records[0])["fundus"]["groundtruth"]
    assert gt["annotation_complete"] is False
    assert gt["hard"]["pmw"]["status"] == "unannotated"
    assert gt["hard"]["pmw"]["annotation_valid"] is False
    assert gt["hard"]["dv"]["annotation_valid"] is True
    assert gt["hard"]["dh"]["annotation_valid"] is True


def test_default_disabled_does_not_open_rgb_or_construct_readout(module_factory, monkeypatch, tmp_path):
    implementation = importlib.import_module("src.models.seg_module")
    def forbidden(*args, **kwargs):
        raise AssertionError("A disabled fundus export must not read RGB or build geometry")
    monkeypatch.setattr(implementation, "build_polar_readout", forbidden)
    model = module_factory(polar=False)
    batch = _batch(tmp_path)
    batch["image_path"] = [str(tmp_path / "does-not-exist.png")]
    monkeypatch.setattr(Image, "open", forbidden)
    model.on_test_epoch_start()
    model.test_step(batch, 0)
    assert model._quantification_readout() is None
    assert model._quantification_overlay_writer is None
    assert model.test_records[0]["overlay_fundus_path"] == ""
    assert model.test_records[0]["overlay_gt_fundus_path"] == ""
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("orig_size", [True, False])
def test_rgb_size_is_verified_or_inferred_through_test_step(module_factory, tmp_path, orig_size):
    model = module_factory(fundus="prediction")
    model.on_test_epoch_start()
    batch = _batch(tmp_path, orig_size=orig_size)
    if orig_size:
        batch["orig_size"] = [torch.tensor([80]), torch.tensor([48])]
        with pytest.raises(ValueError, match="original_size|size|shape|dimension"):
            model.test_step(batch, 0)
        assert not list((tmp_path / "exports").rglob("*.png"))
    else:
        model.test_step(batch, 0)
        assert _metadata(model.test_records[0])["fundus"]["original_shape"] == [48, 80]


@pytest.mark.parametrize("odppa", [False, True])
def test_fundus_paths_survive_ordinary_odppa_quantification_csv_and_reference_parser(
    module_factory, monkeypatch, tmp_path, odppa,
):
    from src.evaluation.evaluator import OVERLAY_PATH_FIELDS, SegEvaluator
    from src.evaluation.quantification_stats import read_reference_csv
    assert OVERLAY_PATH_FIELDS[-2:] == ["overlay_fundus_path", "overlay_gt_fundus_path"]
    model = module_factory(fundus="both", adapter=True)
    model.on_test_epoch_start()
    model.test_step(_batch(tmp_path), 0)
    model.cfg.eval.odppa.enabled = odppa
    if odppa:
        statistics = importlib.import_module("src.evaluation.odppa_stats")
        monkeypatch.setattr(statistics, "summarize_records", lambda *args, **kwargs: [])
        model.test_records[0].update({
            "reference_complete": True, "od_reference_valid": True,
            "od_pred_empty": False, "od_hd95_valid": True,
            "ppa_pred_empty": False, "ppa_hd95_valid": True,
        })
    exported = SegEvaluator(model.cfg, split="test", run_name="tiny").write(model.test_records, tmp_path / "csv")
    for name in ("per_case", "quantification_per_case"):
        with Path(exported[name]).open(newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames[-2:] == ["overlay_fundus_path", "overlay_gt_fundus_path"]
            row = next(reader)
        parsed = read_reference_csv(exported[name])[0]
        for key in ("overlay_fundus_path", "overlay_gt_fundus_path"):
            assert row[key] == model.test_records[0][key]
            assert parsed[key] == row[key]
            assert Path(parsed[key]).is_file()


def _cli_args(tmp_path, *extra):
    return [
        "predict_masks.py", "--ckpt", "unused.ckpt", "--model_config", "tiny",
        "--dataset", "tiny_polar", "--dataset_short", "sample", "--model_name", "tiny",
        "--output_root", str(tmp_path / "cli"), "--save_masks", "none", *extra,
    ]


@pytest.mark.parametrize("mode,explicit", [("prediction", True), ("groundtruth", True), ("both", False)])
def test_cli_fundus_only_honors_flags_or_config_and_uses_correct_forward(
    module_factory, monkeypatch, tmp_path, mode, explicit,
):
    entry = importlib.import_module("predict_masks")
    adapter = mode in {"prediction", "both"}
    cfg = module_factory(fundus="none" if explicit else mode, adapter=adapter,
                         source="quantification").cfg
    if not explicit:
        cfg.eval.fundus_overlay_alpha = 0.25
    monkeypatch.setattr(entry, "build_cfg", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    created = []
    def load(*args, cfg, **kwargs):
        model = module_factory(cfg=cfg)
        created.append((model, _calls(monkeypatch, model._quantification_readout())))
        return model
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", load)
    batch = _batch(tmp_path, complete=False)
    class DataModule:
        def __init__(self, cfg, eval_split):
            assert eval_split == "test"
        def setup(self, stage):
            assert stage == "test"
        def test_dataloader(self):
            return [batch]
    monkeypatch.setattr(entry, "SegDataModule", DataModule)
    flags = ["--save_fundus_overlays", mode, "--fundus_overlay_alpha", "0.25"] if explicit else []
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path, *flags))
    entry.main()
    model, calls = created[0]
    assert model.model.calls == [adapter]
    assert calls == {"hard": 2 if mode == "both" else 1, "soft": 0}
    if adapter:
        assert model.quant_head.calls == 1
    out = tmp_path / "cli" / "tiny"
    metadata_dir = out / "overlay_metadata" / "sample"
    metadata = json.loads((metadata_dir / "001.json").read_text())
    assert (metadata_dir / "config.yaml").is_file()
    assert metadata["fundus"]["mode"] == mode
    assert metadata["fundus"]["alpha"] == 0.25
    if mode in {"prediction", "both"}:
        assert (out / "overlays_fundus" / "sample" / "001.png").is_file()
        assert metadata["fundus"]["prediction_source"] == "quant_logits"
        assert metadata["fundus"]["prediction"]["hard"]["raw"]["pmw"] == 5.0
    if mode in {"groundtruth", "both"}:
        assert (out / "overlays_gt_fundus" / "sample" / "001.png").is_file()
        assert metadata["fundus"]["groundtruth"]["annotation_complete"] is False
        assert metadata["fundus"]["groundtruth"]["hard"]["raw"]["pmw"] == 4.0
    assert not (out / "masks_polar").exists()
    assert not (out / "masks_cart").exists()
    assert not (out / "overlays_polar").exists()
    assert not (out / "overlays_cartesian").exists()


def test_cli_all_exports_off_returns_before_loading_model_or_data(module_factory, monkeypatch, tmp_path):
    entry = importlib.import_module("predict_masks")
    cfg = module_factory().cfg
    monkeypatch.setattr(entry, "build_cfg", lambda *args, **kwargs: cfg)
    def forbidden(*args, **kwargs):
        raise AssertionError("All disabled exports must return before model/data loading")
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", forbidden)
    monkeypatch.setattr(entry, "SegDataModule", forbidden)
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path))
    entry.main()
    assert not (tmp_path / "cli").exists()


@pytest.mark.parametrize("mode", ["prediction", "groundtruth"])
def test_evaluate_fundus_only_with_csv_disabled_writes_images_and_resolved_config(
    module_factory, monkeypatch, tmp_path, capsys, mode,
):
    entry = importlib.import_module("evaluate")
    model = module_factory(fundus=mode)
    batch = _batch(tmp_path)
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(entry, "SegDataModule", lambda *args, **kwargs: object())
    monkeypatch.setattr(entry, "TensorBoardLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(entry, "CSVLogger", lambda *args, **kwargs: object())
    class Trainer:
        is_global_zero = True
        def __init__(self, **kwargs):
            pass
        def test(self, model, datamodule):
            model.on_test_epoch_start()
            model.test_step(batch, 0)
            return [{"test/dice": 1.0}]
    monkeypatch.setattr(entry.pl, "Trainer", Trainer)
    entry.main.__wrapped__(model.cfg)
    out = tmp_path / "exports"
    assert (out / "config.yaml").is_file()
    assert not (out / "per_case.csv").exists()
    destination = out / ("overlays_fundus" if mode == "prediction" else "overlays_gt_fundus")
    assert (destination / "001.png").is_file()
    assert str(destination) in capsys.readouterr().out


def test_hydra_config_declares_fundus_defaults_and_accepts_normal_overrides():
    hydra = pytest.importorskip("hydra")
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        default = hydra.compose(config_name="train")
        selected = hydra.compose(config_name="train", overrides=[
            "polar=enabled", "eval.save_fundus_overlays=both", "eval.fundus_overlay_alpha=0.25",
            "eval.save_predictions=none", "eval.save_quantification_overlays=none",
        ])
    assert default.eval.save_fundus_overlays == "none"
    assert default.eval.fundus_overlay_alpha == 0.0
    assert selected.eval.save_fundus_overlays == "both"
    assert selected.eval.fundus_overlay_alpha == 0.25


def test_cli_parser_exposes_independent_fundus_options(tmp_path):
    entry = importlib.import_module("predict_masks")
    parser = entry.build_parser()
    default = parser.parse_args(_cli_args(tmp_path)[1:])
    assert default.save_fundus_overlays is None
    assert default.fundus_overlay_alpha is None
    parsed = parser.parse_args(_cli_args(tmp_path, "--save_fundus_overlays", "groundtruth",
                                         "--fundus_overlay_alpha", "0.4")[1:])
    assert parsed.save_fundus_overlays == "groundtruth"
    assert parsed.fundus_overlay_alpha == 0.4
    with pytest.raises(SystemExit):
        parser.parse_args(_cli_args(tmp_path, "--save_fundus_overlays", "cartesian")[1:])
