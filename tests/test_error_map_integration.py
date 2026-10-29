"""Real model/export integration for independent hard FP/FN visualizations.

Small deterministic networks and loader/Trainer boundaries replace expensive
training machinery. SegModule selection, PNG/JSON output and CSV code are real.
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
cv2 = pytest.importorskip("cv2")
omegaconf = pytest.importorskip("omegaconf")

from src.evaluation.error_maps import ERROR_MAP_PATH_FIELDS


def _mask(polar=False, extra_width=0):
    if polar:
        labels = torch.zeros(16, 32, dtype=torch.long)
        labels[:, :7] = 1
        labels[:, 7:10 + extra_width] = 2
    else:
        labels = torch.zeros(32, 32, dtype=torch.long)
        labels[8:20, 8:18] = 1
        labels[8:20, 18:21 + extra_width] = 2
    return labels


def _logits(labels):
    return torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 16


class _Network(torch.nn.Module):
    def __init__(self, polar, count):
        super().__init__()
        labels = _mask(polar).unsqueeze(0).repeat(count, 1, 1)
        self.register_buffer("logits", _logits(labels))
        self.register_buffer("feature", torch.ones(count, 4, *labels.shape[-2:]))
        self.calls = []

    def forward(self, images, return_features=False):
        self.calls.append(bool(return_features))
        return {"seg_logits": self.logits, "features": {"decoder_final": self.feature}}


class _Adapter(torch.nn.Module):
    def __init__(self, polar, count):
        super().__init__()
        self.register_buffer("logits", _logits(_mask(polar, 2).unsqueeze(0).repeat(count, 1, 1)))
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

    def make(*, mode="none", fundus=False, source="seg_logits", polar=False,
             adapter=False, count=1, output_size="original", cfg=None):
        if cfg is None:
            size = 64 if polar else 32
            cfg = omegaconf.OmegaConf.create({
                "dataset": {
                    "name": "tiny_polar" if polar else "tiny_cart", "num_classes": 3,
                    "mask_mode": "index", "class_names": ["background", "OD", "PPA"],
                    "foreground_classes": [1, 2],
                },
                "model": {"arch": "tiny", "decoder_channels": [4]},
                "train": {"img_size": size},
                "polar": {"enabled": polar, "geometry": {
                    "center": [27.25, 31.5] if polar else [13.25, 17.5],
                    "max_radius": 32.0, "angle_bins": 16, "radius_bins": 32,
                    "pt_type": "corner", "mode": "linear",
                }, "eval": {"metric_space": "cart"}},
                "quantification": {
                    "enabled": adapter, "adapter_enabled": adapter, "weight": 0.1,
                    "dv_mode": "max_center_chord", "denominator_floor_px": 1.0,
                    "od_class": 1, "ppa_class": 2,
                },
                "eval": {
                    "save_csv": False, "save_predictions": "none",
                    "save_quantification_overlays": "none", "save_quantification_gt_overlays": "none",
                    "save_fundus_overlays": "none", "quantification_overlay_legend": False,
                    "save_error_maps": mode, "save_error_fundus": fundus,
                    "error_map_classes": "both", "error_map_source": source,
                    "error_map_alpha": 0.6, "error_map_legend": False,
                    "prediction_cartesian_size": output_size,
                    "output_dir": str(tmp_path / "exports"),
                    "odppa": {"enabled": False, "bootstrap_samples": 8, "permutation_samples": 8},
                },
                "logging": {"save_dir": str(tmp_path), "name": "tiny", "logger": "tensorboard"},
                "hardware": {"accelerator": "cpu", "devices": 1, "num_nodes": 1, "strategy": "auto"},
                "split": "test", "checkpoint": "unused.ckpt",
            })
        polar = bool(cfg.polar.enabled)
        network = _Network(polar, count)
        monkeypatch.setattr(implementation, "build_model", lambda *args: network)
        monkeypatch.setattr(implementation, "build_loss", lambda *args: _Loss())
        model = implementation.SegModule(cfg).eval()
        monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
        if bool(cfg.quantification.get("adapter_enabled", False)):
            model.quant_head = _Adapter(polar, count)
        return model

    return make


def _native(labels, cfg, shape):
    array = labels.numpy().astype(np.uint8)
    if bool(cfg.polar.enabled):
        g = cfg.polar.geometry
        size = int(cfg.train.img_size)
        array = cv2.warpPolar(
            array, (size, size), tuple(g.center), float(g.max_radius),
            cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.WARP_FILL_OUTLIERS | cv2.INTER_NEAREST,
        )
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _batch(tmp_path, cfg, count=1, *, complete=True, orig_size=True):
    samples = []
    for index, shape in enumerate([(48, 80), (64, 40)][:count]):
        labels = _mask(bool(cfg.polar.enabled), 1)
        native = _native(labels, cfg, shape)
        image_path, mask_path = tmp_path / f"rgb_{index}.png", tmp_path / f"gt_{index}.png"
        Image.new("RGB", (shape[1], shape[0]), (31 + index, 69, 113)).save(image_path)
        Image.fromarray(native).save(mask_path)
        sample = {
            "image": torch.zeros(3, *labels.shape), "mask": labels,
            "case_id": f"{index + 1:03d}", "image_path": str(image_path), "mask_path": str(mask_path),
            "reference_complete": str(complete).lower(),
        }
        if orig_size:
            sample["orig_size"] = shape
        samples.append(sample)
    return torch.utils.data.default_collate(samples)


def _metadata(row):
    return json.loads(Path(row["error_metadata_path"]).read_text())


def _capture(monkeypatch, writer):
    saves = []
    original = writer.save
    def save(prediction, case_id, **kwargs):
        saves.append((prediction.copy(), case_id, kwargs))
        return original(prediction, case_id, **kwargs)
    monkeypatch.setattr(writer, "save", save)
    return saves


@pytest.mark.parametrize("mode,fundus", [("both", False), ("none", True), ("cartesian", True)])
def test_cartesian_error_only_needs_no_quantification_readout_and_routes_two_cases(
    module_factory, monkeypatch, tmp_path, mode, fundus,
):
    implementation = importlib.import_module("src.models.seg_module")
    def forbidden(*args, **kwargs):
        raise AssertionError("Error-only export must not build or compute quantification readouts")
    monkeypatch.setattr(implementation, "build_polar_readout", forbidden)
    monkeypatch.setattr(implementation, "compute_quantification_diagnostics", forbidden)
    model = module_factory(mode=mode, fundus=fundus, count=2)
    assert model._quantification_readout() is None
    assert not model.quant_test_metrics
    model.on_test_epoch_start()
    saved = _capture(monkeypatch, model._error_map_writer)
    model.test_step(_batch(tmp_path, model.cfg, count=2), 0)
    assert model.model.calls == [False]
    assert model._prediction_writer is model._quantification_overlay_writer is None
    assert len(model.test_records) == len(saved) == 2
    for index, (row, save, shape) in enumerate(zip(model.test_records, saved, ((48, 80), (64, 40)), strict=True)):
        prediction, case_id, arguments = save
        assert case_id == f"{index + 1:03d}"
        assert tuple(arguments["original_size"]) == shape
        assert Path(arguments["mask_path"]) == tmp_path / f"gt_{index}.png"
        assert Path(arguments["image_path"]) == tmp_path / f"rgb_{index}.png"
        np.testing.assert_array_equal(prediction, _mask())
        np.testing.assert_array_equal(arguments["groundtruth_mask"], _mask(extra_width=1))
        meta = _metadata(row)
        assert meta["actual_source"] == "seg_logits"
        assert meta["original_shape"] == list(shape)
        assert not any(key.startswith("quant_") for key in row)
        assert row["pred_path"] == row["pred_polar_path"] == row["pred_cartesian_path"] == ""
        assert row["overlay_fundus_path"] == row["overlay_gt_fundus_path"] == ""
        for label in ("od", "ppa"):
            for space, selected in (("polar", mode == "both"),
                                    ("cartesian", mode in {"cartesian", "both"}), ("fundus", fundus)):
                path = row[f"error_{label}_{space}_path"]
                assert bool(path) is selected
                if selected:
                    with Image.open(path) as image:
                        assert image.mode == "RGB"
                        assert (image.height, image.width) == ((16, 32) if space == "polar" else shape)
        assert all(not isinstance(value, (np.ndarray, torch.Tensor)) for value in row.values())


@pytest.mark.parametrize("adapter,source,actual_source,extra", [
    (False, "quantification", "seg_logits", 0),
    (True, "seg_logits", "seg_logits", 0),
    (True, "quantification", "quant_logits", 2),
])
def test_error_prediction_source_uses_one_matching_forward_and_preserves_saved_segmentation(
    module_factory, monkeypatch, tmp_path, adapter, source, actual_source, extra,
):
    model = module_factory(mode="polar", polar=True, adapter=adapter, source=source)
    model.cfg.eval.save_predictions = "polar"
    model.on_test_epoch_start()
    saved = _capture(monkeypatch, model._error_map_writer)
    model.test_step(_batch(tmp_path, model.cfg), 0)
    assert model.model.calls == [adapter]
    if adapter:
        assert model.quant_head.calls == 1
    np.testing.assert_array_equal(saved[0][0], _mask(True, extra))
    assert saved[0][2]["prediction_source"] == actual_source
    meta = _metadata(model.test_records[0])
    assert meta["selected_source"] == source
    assert meta["actual_source"] == actual_source
    ppa = meta["spaces"]["polar"]["classes"]["ppa"]["counts"]
    assert ppa["fp"] == (16 if extra == 2 else 0)
    assert ppa["fn"] == (0 if extra == 2 else 16)
    with Image.open(model.test_records[0]["pred_polar_path"]) as image:
        np.testing.assert_array_equal(image, _mask(True))


@pytest.mark.parametrize("metadata_source", ["batch", "csv"])
def test_partial_ppa_is_omitted_and_csv_completeness_takes_precedence(
    module_factory, tmp_path, metadata_source,
):
    model = module_factory(mode="cartesian", fundus=True)
    if metadata_source == "csv":
        metadata_path = tmp_path / "reference.csv"
        metadata_path.write_text("case_id,reference_complete\n001,false\n")
        model.cfg.eval.odppa.metadata_csv = str(metadata_path)
        model = module_factory(cfg=model.cfg)
    model.on_test_epoch_start()
    model.test_step(_batch(tmp_path, model.cfg, complete=metadata_source == "csv"), 0)
    row = model.test_records[0]
    meta = _metadata(row)
    assert meta["reference_complete"] is False
    for space in ("cartesian", "fundus"):
        assert row[f"error_ppa_{space}_path"] == ""
        assert Path(row[f"error_od_{space}_path"]).is_file()
        ppa = meta["spaces"][space]["classes"]["ppa"]
        assert ppa["status"] == "not_evaluable"
        assert ppa["counts"] is None


def test_default_disabled_preserves_all_new_empty_paths_without_file_reads(module_factory, monkeypatch, tmp_path):
    model = module_factory()
    batch = _batch(tmp_path, model.cfg)
    batch["image_path"] = ["missing-rgb.png"]
    batch["mask_path"] = ["missing-gt.png"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled error-map export must not open metadata paths")
    monkeypatch.setattr(Image, "open", forbidden)
    model.on_test_epoch_start()
    model.test_step(batch, 0)
    assert model.model.calls == [False]
    assert model._error_map_writer is None
    assert all(model.test_records[0][key] == "" for key in ERROR_MAP_PATH_FIELDS)
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("output_size", ["input", "original"])
def test_even_input_exports_validate_collated_original_size(module_factory, tmp_path, output_size):
    model = module_factory(mode="cartesian", output_size=output_size)
    model.on_test_epoch_start()
    batch = _batch(tmp_path, model.cfg)
    batch["orig_size"] = [torch.tensor([80]), torch.tensor([48])]
    with pytest.raises(ValueError, match="size|shape|dimension"):
        model.test_step(batch, 0)
    assert not list((tmp_path / "exports").rglob("*.png"))


def test_missing_size_is_derived_from_native_gt_without_opening_unused_rgb(module_factory, monkeypatch, tmp_path):
    model = module_factory(mode="cartesian")
    model.on_test_epoch_start()
    batch = _batch(tmp_path, model.cfg, orig_size=False)
    batch["image_path"] = [str(tmp_path / "missing-rgb.png")]
    opened = []
    original = Image.open
    def open_image(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", open_image)
    model.test_step(batch, 0)
    assert _metadata(model.test_records[0])["original_shape"] == [48, 80]
    assert str(tmp_path / "gt_0.png") in opened
    assert str(tmp_path / "missing-rgb.png") not in opened


def test_native_gt_failure_is_not_replaced_by_transformed_batch_gt(module_factory, tmp_path):
    model = module_factory(mode="cartesian")
    model.on_test_epoch_start()
    batch = _batch(tmp_path, model.cfg)
    batch["mask_path"] = [str(tmp_path / "missing.png")]
    with pytest.raises((ValueError, OSError)):
        model.test_step(batch, 0)
    assert not list((tmp_path / "exports").rglob("*.png"))


def test_batch_helper_allows_native_only_comparison_but_polar_needs_batch_gt(module_factory):
    model = module_factory(mode="cartesian")
    logits = model.model.logits
    payload = model._error_map_batch(logits, logits, "seg_logits", target_masks=None)
    assert payload["groundtruth_mask"] is None
    np.testing.assert_array_equal(payload["mask"][0], _mask())
    model = module_factory(mode="polar")
    with pytest.raises(ValueError, match="target|GT|groundtruth"):
        model._error_map_batch(logits, logits, "seg_logits", target_masks=None)


@pytest.mark.parametrize("odppa", [False, True])
def test_error_paths_roundtrip_in_ordinary_odppa_and_quantification_case_csvs(
    module_factory, monkeypatch, tmp_path, odppa,
):
    from src.evaluation.evaluator import EXPORT_PATH_FIELDS, SegEvaluator
    from src.evaluation.quantification_stats import read_reference_csv
    model = module_factory(mode="cartesian", fundus=True, polar=True, adapter=True)
    model.on_test_epoch_start()
    model.test_step(_batch(tmp_path, model.cfg), 0)
    model.cfg.eval.odppa.enabled = odppa
    if odppa:
        statistics = importlib.import_module("src.evaluation.odppa_stats")
        monkeypatch.setattr(statistics, "summarize_records", lambda *args, **kwargs: [])
        model.test_records[0].update({
            "reference_complete": True, "od_reference_valid": True,
            "od_pred_empty": False, "od_hd95_valid": True,
            "ppa_pred_empty": False, "ppa_hd95_valid": True,
        })
    files = SegEvaluator(model.cfg, split="test", run_name="tiny").write(model.test_records, tmp_path / "csv")
    for name in ("per_case", "quantification_per_case"):
        with Path(files[name]).open(newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames[-len(ERROR_MAP_PATH_FIELDS):] == list(ERROR_MAP_PATH_FIELDS)
            assert reader.fieldnames[-len(EXPORT_PATH_FIELDS):] == EXPORT_PATH_FIELDS
            row = next(reader)
        parsed = read_reference_csv(files[name])[0]
        assert parsed["case_id"] == "001"
        for key in ERROR_MAP_PATH_FIELDS:
            assert row[key] == parsed[key] == model.test_records[0][key]
            if row[key]:
                assert Path(row[key]).is_file()


def _cli_args(tmp_path, *extra):
    return [
        "predict_masks.py", "--ckpt", "unused.ckpt", "--model_config", "tiny",
        "--dataset", "tiny_cart", "--dataset_short", "sample", "--model_name", "tiny",
        "--output_root", str(tmp_path / "cli"), "--save_masks", "none", *extra,
    ]


@pytest.mark.parametrize("mode,fundus,adapter,explicit", [
    ("polar", False, True, True), ("none", True, False, True), ("cartesian", True, False, False),
])
def test_cli_error_only_flags_and_config_keep_source_and_use_one_forward(
    module_factory, monkeypatch, tmp_path, mode, fundus, adapter, explicit,
):
    entry = importlib.import_module("predict_masks")
    cfg = module_factory(polar=adapter, adapter=adapter).cfg
    classes = "od" if fundus and mode == "none" else "both"
    if not explicit:
        cfg.eval.update(save_error_maps=mode, save_error_fundus=fundus, error_map_source="quantification",
                        error_map_classes=classes, error_map_alpha=0.4)
    monkeypatch.setattr(entry, "build_cfg", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    created = []
    def load(*args, cfg, **kwargs):
        model = module_factory(cfg=cfg)
        created.append(model)
        return model
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", load)
    batch = _batch(tmp_path, cfg)
    class DataModule:
        def __init__(self, cfg, eval_split):
            assert eval_split == "test"
        def setup(self, stage):
            assert stage == "test"
        def test_dataloader(self):
            return [batch]
    monkeypatch.setattr(entry, "SegDataModule", DataModule)
    flags = ["--save_error_maps", mode, "--save_error_fundus", str(fundus).lower(),
             "--error_map_source", "quantification", "--error_map_classes", classes,
             "--error_map_alpha", "0.4"] if explicit else []
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path, *flags))
    entry.main()
    model = created[0]
    assert model.model.calls == [adapter]
    if adapter:
        assert model.quant_head.calls == 1
    else:
        assert model._quantification_readout() is None
    out = tmp_path / "cli" / "tiny"
    metadata_dir = out / "error_metadata" / "sample"
    assert (metadata_dir / "config.yaml").is_file()
    meta = json.loads((metadata_dir / "001.json").read_text())
    assert meta["selected_source"] == "quantification"
    assert meta["actual_source"] == ("quant_logits" if adapter else "seg_logits")
    assert meta["mode"] == mode
    assert meta["save_fundus"] is fundus
    assert meta["classes"] == classes
    if fundus:
        assert meta["spaces"]["fundus"]["alpha"] == 0.4
    for space in meta["spaces"]:
        for label in meta["spaces"][space]["classes"]:
            assert Path(meta["paths"][f"error_{label}_{space}_path"]).is_file()
    assert not (out / "masks_polar").exists()
    assert not (out / "overlays_polar").exists()
    assert not (out / "overlays_fundus").exists()


def test_cli_csv_reference_status_overrides_batch_for_error_maps(module_factory, monkeypatch, tmp_path):
    entry = importlib.import_module("predict_masks")
    cfg = module_factory(mode="cartesian").cfg
    metadata_path = tmp_path / "reference.csv"
    metadata_path.write_text("case_id,reference_complete\n001,false\n")
    cfg.eval.odppa.metadata_csv = str(metadata_path)
    monkeypatch.setattr(entry, "build_cfg", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", lambda *args, cfg, **kwargs: module_factory(cfg=cfg))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batch = _batch(tmp_path, cfg, complete=True)
    class DataModule:
        def __init__(self, cfg, eval_split):
            pass
        def setup(self, stage):
            pass
        def test_dataloader(self):
            return [batch]
    monkeypatch.setattr(entry, "SegDataModule", DataModule)
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path))
    entry.main()
    path = tmp_path / "cli" / "tiny" / "error_metadata" / "sample" / "001.json"
    meta = json.loads(path.read_text())
    assert meta["reference_complete"] is False
    assert meta["spaces"]["cartesian"]["classes"]["ppa"]["counts"] is None
    assert meta["paths"]["error_ppa_cartesian_path"] == ""


def test_cli_default_disabled_exits_without_model_data_or_output(module_factory, monkeypatch, tmp_path):
    entry = importlib.import_module("predict_masks")
    cfg = module_factory().cfg
    monkeypatch.setattr(entry, "build_cfg", lambda *args, **kwargs: cfg)
    def forbidden(*args, **kwargs):
        raise AssertionError("All exports are disabled")
    monkeypatch.setattr(entry.SegModule, "load_from_checkpoint", forbidden)
    monkeypatch.setattr(entry, "SegDataModule", forbidden)
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path))
    entry.main()
    assert not (tmp_path / "cli").exists()


@pytest.mark.parametrize("mode,fundus", [("cartesian", False), ("none", True)])
def test_evaluate_error_only_and_save_csv_false_still_save_resolved_configuration(
    module_factory, monkeypatch, tmp_path, capsys, mode, fundus,
):
    entry = importlib.import_module("evaluate")
    model = module_factory(mode=mode, fundus=fundus)
    batch = _batch(tmp_path, model.cfg)
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
    root = tmp_path / "exports"
    assert (root / "config.yaml").is_file()
    assert not (root / "per_case.csv").exists()
    directory = root / ("errors_fundus" if fundus else "errors_cartesian")
    assert (directory / "od" / "001.png").is_file()
    assert str(directory) in capsys.readouterr().out


def test_hydra_and_cli_declare_independent_defaults_without_overwriting_config(tmp_path):
    hydra = pytest.importorskip("hydra")
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        defaults = hydra.compose(config_name="train")
        selected = hydra.compose(config_name="train", overrides=[
            "eval.save_error_maps=both", "eval.save_error_fundus=true", "eval.error_map_classes=ppa",
            "eval.error_map_source=quantification", "eval.error_map_alpha=0.4",
        ])
    assert defaults.eval.save_error_maps == "none"
    assert defaults.eval.save_error_fundus is False
    assert defaults.eval.error_map_classes == "both"
    assert defaults.eval.error_map_source == "seg_logits"
    assert defaults.eval.error_map_alpha == 0.6
    assert defaults.eval.error_map_legend is True
    assert selected.eval.save_error_maps == "both"
    assert selected.eval.save_error_fundus is True
    entry = importlib.import_module("predict_masks")
    parser = entry.build_parser()
    args = parser.parse_args(_cli_args(tmp_path)[1:])
    assert args.save_error_maps is args.save_error_fundus is args.error_map_classes is None
    assert args.error_map_source is args.error_map_alpha is None
    with pytest.raises(SystemExit):
        parser.parse_args(_cli_args(tmp_path, "--save_error_fundus", "maybe")[1:])
