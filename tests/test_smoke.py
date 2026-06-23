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
