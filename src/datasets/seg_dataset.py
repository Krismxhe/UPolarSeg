"""
Generic 2D medical image segmentation dataset.

Expected directory structure:
    root/
    ├── train/
    │   ├── images/       ← RGB images (.png or .jpg)
    │   └── {mask_dir}/   ← segmentation masks (.png)
    ├── val/
    │   ├── images/
    │   └── {mask_dir}/
    └── test/
        ├── images/
        └── {mask_dir}/
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SegDataset(Dataset):
    """
    Args:
        root      : Path to dataset root.
        split     : "train", "val", or "test".
        mask_dir  : Name of mask subdirectory (e.g. "masks_semantic").
        mask_mode : "index"  → pixel values are class indices {0, 1, 2, ...}
                    "binary" → pixel values are {0, 255}, converted to {0, 1}.
        transforms: Albumentations Compose pipeline (or None).
    """

    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    def __init__(self, root, split, mask_dir, mask_mode='index', transforms=None):
        self.img_dir  = Path(root) / split / 'images'
        self.mask_dir = Path(root) / split / mask_dir
        self.mask_mode = mask_mode
        self.transforms = transforms

        self.images = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        # mask has the same stem as the image, always stored as .png
        mask_path = self.mask_dir / (img_path.stem + '.png')

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for image '{img_path.name}': expected '{mask_path}'"
            )

        image = np.array(Image.open(img_path).convert('RGB'))
        mask  = np.array(Image.open(mask_path))

        # Record original spatial size (H, W) BEFORE any resizing transform,
        # so downstream evaluation can restore predictions to native resolution.
        orig_h, orig_w = image.shape[:2]

        if self.mask_mode == 'binary':
            mask = (mask > 127).astype(np.uint8)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']   # C×H×W float32 tensor (via ToTensorV2)
            mask  = augmented['mask']    # H×W uint8 tensor

        return {
            'image': image,
            'mask': mask.long(),
            'case_id': img_path.stem,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'orig_size': (int(orig_h), int(orig_w)),
        }


# ─────────────────────────────────────────────────────────────────────────────

class SegDataModule(pl.LightningDataModule):
    """Wraps SegDataset for use with PyTorch Lightning Trainer."""

    def __init__(self, cfg, eval_split: str = 'test'):
        super().__init__()
        self.cfg = cfg
        self.eval_split = eval_split  # which split trainer.test() evaluates

    def setup(self, stage=None):
        from src.transforms.build_transforms import build_transforms

        ds = self.cfg.dataset

        train_tf = build_transforms(self.cfg, split='train')
        eval_tf  = build_transforms(self.cfg, split='val')

        kwargs = dict(
            root=ds.root,
            mask_dir=ds.mask_dir,
            mask_mode=ds.mask_mode,
        )
        self.train_ds = SegDataset(split='train', transforms=train_tf, **kwargs)
        self.val_ds   = SegDataset(split='val',   transforms=eval_tf,  **kwargs)
        self.test_ds  = SegDataset(split='test',  transforms=eval_tf,  **kwargs)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.train.num_workers > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        ds_map = {'train': self.train_ds, 'val': self.val_ds, 'test': self.test_ds}
        if self.eval_split not in ds_map:
            raise ValueError(f"eval_split must be one of {list(ds_map)}, got '{self.eval_split}'")
        return self._loader(ds_map[self.eval_split], shuffle=False)
