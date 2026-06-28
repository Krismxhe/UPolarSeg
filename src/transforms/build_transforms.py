"""
Build an Albumentations augmentation pipeline from a Hydra config.

Each transform in the augmentation config has an "enabled" flag.
Disabled transforms are simply skipped — no code changes needed.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from .polar_transforms import PolarTransform

def _add_polar(cfg, pipeline, aug, split: str, img_size: int) -> None:
    
    radius_bins = cfg.polar.geometry.get("radius_bins", None)
    angle_bins = cfg.polar.geometry.get("angle_bins", None)
    max_radius = cfg.polar.geometry.get("max_radius", None)
    center = cfg.polar.geometry.get("center", None)
    pt_type = cfg.polar.geometry.get("pt_type", None)
    mode = cfg.polar.geometry.get("mode", None)
    pipeline.append(
        PolarTransform(
            radius_bins=radius_bins or img_size,
            angle_bins=angle_bins or img_size,
            center=center,
            max_radius=max_radius,
            mode=mode,
            pt_type=pt_type,
        )
    )


def build_transforms(cfg, split: str = 'train') -> A.Compose:
    """
    Args:
        cfg   : Full Hydra config (uses cfg.augmentation and cfg.train.img_size).
        split : "train" applies all enabled transforms;
                "val" / "test" only applies Resize + Normalize.

    Returns:
        albumentations.Compose pipeline.
    """
        
    aug      = cfg.augmentation
    img_size = cfg.train.img_size
    mean     = list(aug.normalize.mean)
    std      = list(aug.normalize.std)
    polar    = cfg.polar.enabled
    
    if cfg.polar.enabled:
        h, w = img_size, img_size
        center = ((w-1)/2, (h-1)/2)
        

    pipeline = []

    # ── Train-only transforms ─────────────────────────────────────────────────
    if split == 'train':

        # Spatial ──────────────────────────────────────────────────────────────
        rrc = aug.random_resized_crop
        if rrc.enabled:
            pipeline.append(A.RandomResizedCrop(
                height=img_size, width=img_size,
                scale=tuple(rrc.scale),
            ))
        else:
            pipeline.append(A.Resize(height=img_size, width=img_size))

        hf = aug.horizontal_flip
        if hf.enabled:
            pipeline.append(A.HorizontalFlip(p=hf.p))

        vf = aug.vertical_flip
        if vf.enabled:
            pipeline.append(A.VerticalFlip(p=vf.p))

        rr = aug.random_rotate90
        if rr.enabled:
            pipeline.append(A.RandomRotate90(p=rr.p))

        el = aug.elastic
        if el.enabled:
            pipeline.append(A.ElasticTransform(p=el.p))

        gd = aug.grid_distortion
        if gd.enabled:
            pipeline.append(A.GridDistortion(p=gd.p))

        # Polar run after normal spatial augmentation/resizing,
        if cfg.polar.enabled:
            _add_polar(cfg, pipeline, aug, split=split, img_size=img_size)
        
        # Photometric (image only, mask unaffected) ────────────────────────────
        cj = aug.color_jitter
        if cj.enabled:
            pipeline.append(A.ColorJitter(
                brightness=cj.brightness,
                contrast=cj.contrast,
                saturation=cj.saturation,
                hue=cj.hue,
                p=cj.p,
            ))

        gb = aug.gaussian_blur
        if gb.enabled:
            pipeline.append(A.GaussianBlur(
                sigma_limit=tuple(gb.sigma_limit),
                p=gb.p,
            ))

        gn = aug.gaussian_noise
        if gn.enabled:
            pipeline.append(A.GaussNoise(p=gn.p))
        
    # ── Val / Test: only resize ───────────────────────────────────────────────
    else:
        pipeline.append(A.Resize(height=img_size, width=img_size))
        if cfg.polar.enabled:
            _add_polar(cfg, pipeline, aug, split=split, img_size=img_size)

    # ── Always applied ────────────────────────────────────────────────────────
    pipeline.append(A.Normalize(mean=mean, std=std))
    pipeline.append(ToTensorV2())

    return A.Compose(pipeline)