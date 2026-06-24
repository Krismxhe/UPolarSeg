"""
Phase 0 smoke tests — minimal safety net.

These tests intentionally stay lightweight: they verify that the package
imports, the entry-point scripts compile, the Hydra config composes, the
augmentation pipeline builds, and the demo dataset loads with the *current*
batch contract.

The dataset test is skipped automatically when the demo dataset is not present,
so the suite still passes in a clean checkout / CI without data.
"""

import importlib
import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# demo-dataset lives next to the repo (sibling directory).
DEMO_ROOT = REPO_ROOT.parent / "demo-dataset"
ENTRY_SCRIPTS = ["train.py", "evaluate.py", "predict.py"]


def test_src_modules_import():
    for mod in (
        "src.datasets.seg_dataset",
        "src.models.seg_module",
        "src.transforms.build_transforms",
    ):
        importlib.import_module(mod)


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_entry_scripts_compile(script):
    py_compile.compile(str(REPO_ROOT / script), doraise=True)


def _compose_cfg():
    from hydra import compose, initialize

    # config_path is relative to this test file (tests/ -> ../configs).
    with initialize(config_path="../configs", version_base=None):
        return compose(config_name="train")


def test_experiment_configs_compose():
    from hydra import compose, initialize

    exp_dir = REPO_ROOT / "configs" / "experiment"
    names = sorted(p.stem for p in exp_dir.glob("*.yaml"))
    assert names, "no experiment configs found"
    for name in names:
        with initialize(config_path="../configs", version_base=None):
            cfg = compose(config_name="train", overrides=[f"+experiment={name}"])
        assert str(cfg.logging.name).startswith("exp_")   # pinned run name
        assert "provider" in cfg.model                      # model group resolved
        assert cfg.loss.name in ("dice_ce", "dice_bce")     # loss group resolved


def test_config_composes():
    cfg = _compose_cfg()
    assert cfg.model.arch == "Unet"
    assert cfg.dataset.num_classes == 3
    assert cfg.dataset.mask_mode == "index"


def test_build_transforms_returns_compose():
    import albumentations as A

    from src.transforms.build_transforms import build_transforms

    cfg = _compose_cfg()
    cfg.train.img_size = 64  # keep the smoke test cheap
    for split in ("train", "val"):
        tf = build_transforms(cfg, split=split)
        assert isinstance(tf, A.Compose)


def test_functional_empty_mask_conventions():
    import torch

    from src.metrics.functional import case_class_metrics

    z = torch.zeros(4, 4, dtype=torch.long)
    o = torch.ones(4, 4, dtype=torch.long)

    # pred empty & target empty (class 1 absent in both) → dice=iou=1
    m = case_class_metrics(z, z, [1])[1]
    assert m["dice"] == 1.0 and m["iou"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0

    # pred non-empty & target empty → dice=iou=0
    m = case_class_metrics(o, z, [1])[1]
    assert m["dice"] == 0.0 and m["iou"] == 0.0

    # pred empty & target non-empty → dice=iou=0
    m = case_class_metrics(z, o, [1])[1]
    assert m["dice"] == 0.0 and m["iou"] == 0.0

    # perfect overlap → dice=iou=1
    m = case_class_metrics(o, o, [1])[1]
    assert m["dice"] == 1.0 and m["iou"] == 1.0


def test_functional_no_nan_on_any_combination():
    import math

    import torch

    from src.metrics.functional import case_class_metrics

    z = torch.zeros(2, 2, dtype=torch.long)
    o = torch.ones(2, 2, dtype=torch.long)
    for pred in (z, o):
        for tgt in (z, o):
            for v in case_class_metrics(pred, tgt, [0, 1]).values():
                for key in ("dice", "iou", "precision", "recall"):
                    assert not math.isnan(v[key])


def test_evaluator_writes_three_csvs(tmp_path):
    import csv as _csv

    from src.metrics.evaluator import SegEvaluator

    cfg = _compose_cfg()  # multiclass (3 classes, foreground [1,2])

    def rec(case_id):
        # all classes score perfectly → trivial but exercises aggregation
        per_class = {c: {"dice": 1.0, "iou": 1.0, "precision": 1.0,
                         "recall": 1.0, "support": 10.0} for c in range(cfg.dataset.num_classes)}
        return {"case_id": case_id, "image_path": f"/x/{case_id}.png",
                "mask_path": f"/m/{case_id}.png", "pred_path": "", "per_class": per_class}

    records = [rec("a"), rec("b"), rec("c")]
    evaluator = SegEvaluator(cfg, split="test", run_name="unit", checkpoint="ckpt")
    written = evaluator.write(records, tmp_path)

    for name in ("summary", "per_class", "per_case"):
        assert Path(written[name]).is_file()

    with open(written["per_case"]) as f:
        per_case = list(_csv.DictReader(f))
    assert len(per_case) == len(records)  # one row per case

    with open(written["summary"]) as f:
        summary = list(_csv.DictReader(f))
    assert len(summary) == 1
    assert {"mean_dice", "mean_iou"} <= set(summary[0].keys())
    assert summary[0]["num_cases"] == "3"

    with open(written["per_class"]) as f:
        per_class = list(_csv.DictReader(f))
    # at least the foreground classes are present
    assert len(per_class) == cfg.dataset.num_classes


def test_morphology_basic():
    import torch

    from src.metrics.morphology import area, area_ratio, perimeter

    m = torch.zeros(4, 4, dtype=torch.long)
    m[0:2, 0:2] = 1
    assert area(m == 1) == 4.0
    assert perimeter(m == 1) == 4.0          # all 4 block pixels touch background
    assert area_ratio(m == 1) == 0.25        # 4 / 16
    # empty mask is stable (no NaN, zero metrics)
    assert area(m == 5) == 0.0 and perimeter(m == 5) == 0.0


def test_mask_to_boundary_and_boundary_scores():
    import numpy as np
    import torch

    from src.metrics.boundary_metrics import boundary_scores
    from src.utils.geometry import mask_to_boundary

    solid = torch.ones(6, 6, dtype=torch.long)
    band = mask_to_boundary(solid, width_px=1)
    assert band.dtype == np.bool_ and band.any() and not band.all()

    # identical masks → perfect boundary overlap; empty/empty → 1 (no NaN)
    s = boundary_scores(solid, solid, width_px=1)
    assert s["boundary_dice"] == 1.0 and s["boundary_iou"] == 1.0
    z = torch.zeros(6, 6, dtype=torch.long)
    s0 = boundary_scores(z, z, width_px=1)
    assert s0["boundary_dice"] == 1.0


def test_clinical_metrics_unit_is_pixel():
    import torch

    from src.metrics.clinical_metrics import PIXEL_UNIT, compute_class_clinical

    assert PIXEL_UNIT == "pixel"
    m = torch.zeros(4, 4, dtype=torch.long)
    m[0, 0] = 1
    out = compute_class_clinical(m == 1, ["area", "perimeter", "area_ratio"], total_pixels=16)
    assert out["area"] == 1.0 and out["area_ratio"] == 1 / 16


def test_heads_forward_shapes():
    import torch

    from src.models.heads import BoundaryHead, ClinicalHead

    feat = torch.randn(2, 8, 16, 16)
    assert BoundaryHead(8, 1)(feat).shape == (2, 1, 16, 16)
    assert ClinicalHead(8, 3)(feat).shape == (2, 3)


def test_multitask_loss_seg_only_equals_seg_loss():
    import torch

    from src.losses.factory import build_loss
    from src.losses.multitask_loss import MultiTaskLoss

    cfg = _compose_cfg()
    seg_loss = build_loss(cfg.loss, cfg.dataset)
    mtl = MultiTaskLoss(seg_loss)  # boundary disabled by default

    logits = torch.randn(2, cfg.dataset.num_classes, 16, 16)
    targets = {"mask": torch.randint(0, cfg.dataset.num_classes, (2, 16, 16))}
    total, components = mtl({"seg_logits": logits}, targets)
    assert torch.allclose(total, components["seg_total"])  # seg-only → identical


def test_tasks_disabled_by_default():
    from src.tasks import is_task_enabled

    cfg = _compose_cfg()
    assert is_task_enabled(cfg, "segmentation") is True
    assert is_task_enabled(cfg, "boundary") is False
    assert is_task_enabled(cfg, "clinical") is False


def test_evaluator_clinical_boundary_csvs(tmp_path):
    from omegaconf import open_dict

    from src.metrics.evaluator import SegEvaluator

    cfg = _compose_cfg()  # 3 classes, foreground [1, 2]
    with open_dict(cfg):
        cfg.task.outputs.clinical.enabled = True
        cfg.task.outputs.boundary.enabled = True

    def rec(case_id):
        per_class = {c: {"dice": 1.0, "iou": 1.0, "precision": 1.0,
                         "recall": 1.0, "support": 10.0} for c in range(3)}
        clinical = {"unit": "pixel", "per_class": {
            1: {"area": 5.0, "perimeter": 4.0, "area_ratio": 0.1},
            2: {"area": 3.0, "perimeter": 3.0, "area_ratio": 0.05}}}
        boundary = {"width_px": 3, "per_class": {
            1: {"boundary_dice": 0.8, "boundary_iou": 0.6},
            2: {"boundary_dice": 0.7, "boundary_iou": 0.5}}}
        return {"case_id": case_id, "image_path": "", "mask_path": "", "pred_path": "",
                "per_class": per_class, "clinical": clinical, "boundary": boundary}

    records = [rec("a"), rec("b")]
    written = SegEvaluator(cfg, split="test", run_name="unit").write(records, tmp_path)

    # core CSVs still present
    for name in ("summary", "per_class", "per_case"):
        assert Path(written[name]).is_file()
    # extra CSVs present and one row per case
    import csv as _csv
    for name in ("clinical", "boundary"):
        assert Path(written[name]).is_file()
    with open(written["clinical"]) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2 and rows[0]["unit"] == "pixel"
    assert "area_class_a" in rows[0]


def test_evaluator_no_extra_csv_when_disabled(tmp_path):
    from src.metrics.evaluator import SegEvaluator

    cfg = _compose_cfg()  # tasks default OFF
    rec = {"case_id": "a", "image_path": "", "mask_path": "", "pred_path": "",
           "per_class": {c: {"dice": 1.0, "iou": 1.0, "precision": 1.0,
                             "recall": 1.0, "support": 1.0} for c in range(3)}}
    written = SegEvaluator(cfg, split="test", run_name="unit").write([rec], tmp_path)
    assert "clinical" not in written and "boundary" not in written
    assert not (tmp_path / "clinical_metrics.csv").exists()


def test_config_includes_loss_group():
    cfg = _compose_cfg()
    assert cfg.loss.name == "dice_ce"


def test_build_loss_multiclass_returns_three_terms():
    import torch

    from src.losses.factory import build_loss

    cfg = _compose_cfg()  # multiclass dataset, loss=dice_ce
    loss_fn = build_loss(cfg.loss, cfg.dataset)

    logits = torch.randn(2, cfg.dataset.num_classes, 16, 16, requires_grad=True)
    targets = torch.randint(0, cfg.dataset.num_classes, (2, 16, 16))
    total, dice, aux = loss_fn(logits, targets)
    assert total.ndim == 0 and total.requires_grad
    # default weights are 1.0 → total == dice + aux (old semantics preserved)
    assert torch.allclose(total, dice + aux)


def test_build_loss_fallback_when_cfg_missing():
    """No loss config (e.g. old checkpoint) → backward-compatible default."""
    import torch

    from src.losses.factory import build_loss

    cfg = _compose_cfg()
    loss_fn = build_loss(None, cfg.dataset)  # fallback path
    logits = torch.randn(2, cfg.dataset.num_classes, 8, 8)
    targets = torch.randint(0, cfg.dataset.num_classes, (2, 8, 8))
    total, dice, aux = loss_fn(logits, targets)
    assert torch.allclose(total, dice + aux)


def test_build_loss_rejects_task_mismatch():
    from omegaconf import OmegaConf

    from src.losses.factory import build_loss

    cfg = _compose_cfg()  # multiclass dataset
    # dice_bce on a multiclass task must fail clearly
    bce_cfg = OmegaConf.create({"name": "dice_bce"})
    with pytest.raises(ValueError):
        build_loss(bce_cfg, cfg.dataset)


def test_segmodule_has_no_smp_import():
    """Phase 3: SegModule must not import segmentation_models_pytorch."""
    import src.models.seg_module as seg_module

    assert not hasattr(seg_module, "smp")
    src_text = Path(seg_module.__file__).read_text()
    assert "import segmentation_models_pytorch" not in src_text


def test_normalize_output_tensor_wraps_without_change():
    import torch

    from src.models.outputs import normalize_model_output

    t = torch.randn(2, 3, 8, 8)
    out = normalize_model_output(t)
    assert set(out.keys()) == {"seg_logits"}
    # identity: no copy, no dtype/device/shape change
    assert out["seg_logits"] is t


def test_normalize_output_dict_with_seg_logits_passes_through():
    import torch

    from src.models.outputs import normalize_model_output

    d = {"seg_logits": torch.randn(2, 1, 4, 4), "boundary_logits": torch.randn(2, 1, 4, 4)}
    out = normalize_model_output(d)
    assert out is d  # passed through untouched


def test_normalize_output_dict_missing_seg_logits_raises():
    import torch

    from src.models.outputs import normalize_model_output

    with pytest.raises(ValueError):
        normalize_model_output({"boundary_logits": torch.randn(2, 1, 4, 4)})


def test_normalize_output_rejects_tuple_and_list():
    import torch

    from src.models.outputs import normalize_model_output

    t = torch.randn(2, 1, 4, 4)
    with pytest.raises(TypeError):
        normalize_model_output((t,))
    with pytest.raises(TypeError):
        normalize_model_output([t])


def test_segmodule_model_output_normalizes_to_seg_logits():
    """Old SMP models return a tensor → contract still yields seg_logits unchanged."""
    import torch

    from src.models.outputs import normalize_model_output
    from src.models.seg_module import SegModule

    cfg = _compose_cfg()
    cfg.model.encoder_weights = None  # offline + fast
    cfg.train.img_size = 64
    module = SegModule(cfg).eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        raw = module(x)
        out = normalize_model_output(raw)
    assert isinstance(raw, torch.Tensor)  # SMP baseline still returns a tensor
    assert out["seg_logits"] is raw
    assert out["seg_logits"].shape == (2, cfg.dataset.num_classes, 64, 64)


def test_identity_skip_passthrough():
    import torch

    from src.models.research.modular_unet.skip_modules import build_skip_module

    skip = build_skip_module({"name": "identity"}, channels_by_level=[64, 32])
    x = torch.randn(2, 16, 8, 8)
    out = skip(x, decoder_feature=None, level=0)
    assert out is x  # identity returns the skip unchanged


def test_modular_unet_identity_forward_shape():
    import torch
    from hydra import compose, initialize

    from src.models.factory import build_model

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["model=custom/modular_unet", "model.encoder_weights=null"],
        )
    assert cfg.model.provider == "custom" and cfg.model.name == "modular_unet"

    model = build_model(cfg.model, cfg.dataset).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, cfg.dataset.num_classes, 256, 256)


def test_transunet_forward_shape():
    import torch
    from hydra import compose, initialize

    from src.models.factory import build_model

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "model=custom/transunet",
                "train.img_size=224",
                "model.encoder_weights=null",
                "model.params.num_layers=2",   # keep the test light
            ],
        )
    assert cfg.model.provider == "custom" and cfg.model.name == "transunet"

    model = build_model(cfg.model, cfg.dataset).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, cfg.dataset.num_classes, 224, 224)


def test_transunet_img_size_divisible_check():
    from src.models.baselines.transunet.model import TransUNet

    with pytest.raises(ValueError):
        TransUNet(img_size=225, patch_size=16, encoder_weights=None)


def test_build_model_smp_forward_shape():
    import torch

    from src.models.factory import build_model

    cfg = _compose_cfg()
    cfg.model.encoder_weights = None  # offline + fast: skip imagenet download
    model = build_model(cfg.model, cfg.dataset).eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, cfg.dataset.num_classes, 128, 128)


def test_build_model_rejects_unknown_provider():
    from omegaconf import open_dict

    from src.models.factory import build_model

    cfg = _compose_cfg()
    with open_dict(cfg.model):
        cfg.model.provider = "bogus"
    with pytest.raises(ValueError):
        build_model(cfg.model, cfg.dataset)


def test_unpack_batch_handles_dict_and_tuple():
    import torch

    from src.datasets.batch import unpack_batch

    img = torch.zeros(2, 3, 8, 8)
    msk = torch.zeros(2, 8, 8, dtype=torch.long)

    # dict batch → metadata passed through
    di, dm, meta = unpack_batch({"image": img, "mask": msk, "case_id": ["a", "b"]})
    assert di is img and dm is msk
    assert meta["case_id"] == ["a", "b"]

    # legacy tuple batch → empty metadata
    ti, tm, tmeta = unpack_batch((img, msk))
    assert ti is img and tm is msk
    assert tmeta == {}

    with pytest.raises(TypeError):
        unpack_batch((img, msk, msk))


def _build_demo_dataset():
    from src.datasets.seg_dataset import SegDataset
    from src.transforms.build_transforms import build_transforms

    cfg = _compose_cfg()
    cfg.train.img_size = 64
    tf = build_transforms(cfg, split="val")
    ds = SegDataset(
        root=str(DEMO_ROOT),
        split="train",
        mask_dir=cfg.dataset.mask_dir,
        mask_mode=cfg.dataset.mask_mode,
        transforms=tf,
    )
    return ds, cfg


@pytest.mark.skipif(
    not (DEMO_ROOT / "train" / "images").is_dir(),
    reason="demo-dataset/train not found",
)
def test_demo_dataset_loads():
    import torch

    ds, cfg = _build_demo_dataset()
    assert len(ds) > 0

    # Phase 1 contract: __getitem__ returns a dict.
    sample = ds[0]
    image, mask = sample["image"], sample["mask"]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.long
    assert int(mask.max()) < cfg.dataset.num_classes


@pytest.mark.skipif(
    not (DEMO_ROOT / "train" / "images").is_dir(),
    reason="demo-dataset/train not found",
)
def test_batch_has_metadata():
    ds, _ = _build_demo_dataset()
    sample = ds[0]
    for key in ("image", "mask", "case_id", "image_path", "mask_path", "orig_size"):
        assert key in sample, f"missing metadata key: {key}"
    # orig_size is recorded before resize → native 1024×1024 demo resolution.
    assert sample["orig_size"] == (1024, 1024)
    assert sample["image_path"].endswith(".png")
