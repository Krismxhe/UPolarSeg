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
