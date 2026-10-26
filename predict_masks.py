"""
Generate prediction masks for a checkpoint on one dataset split.

For a polar model: save polar mask + inverse-transformed cart mask.
For a cart model:  save cart mask  + forward-transformed polar mask.

Output structure:
    {output_root}/{model_name}/masks_polar/{dataset_short}/{case_id}.png
    {output_root}/{model_name}/masks_cart/{dataset_short}/{case_id}.png

Usage:
    python predict_masks.py \
        --ckpt /path/to/best.ckpt \
        --model_config baseline/transunet \
        --dataset beijing_1024_cart \
        --dataset_short beijing \
        --split test \
        --model_name TransUNet_cart \
        --output_root /home/ziyue/ziyue/mengxian/pred_mask \
        --img_size 512 \
        --max_radius 362
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import hydra
from PIL import Image
from tqdm import tqdm

from src.datasets.seg_dataset import SegDataModule
from src.models.seg_module import SegModule
from src.datasets.batch import unpack_batch
from src.models.outputs import normalize_model_output
from src.transforms.polar_transforms import cart_to_polar_mask, polar_mask_to_cart


def build_cfg(dataset, model_config, split, img_size, is_polar):
    overrides = [
        f'dataset={dataset}',
        f'model={model_config}',
        f'split={split}',
        f'train.img_size={img_size}',
        'hardware.devices=[0]',
    ]
    if is_polar:
        overrides.append('polar=enabled')
    with hydra.initialize(version_base=None, config_path='configs'):
        cfg = hydra.compose(config_name='train', overrides=overrides)
    return cfg


def resolve_max_radius(cfg, is_polar, fallback):
    """Read max_radius from cfg if available and not None; else use fallback."""
    if not is_polar:
        return float(fallback)
    try:
        mr = cfg.polar.geometry.get('max_radius', None)
    except Exception:
        mr = None
    if mr is None:
        return float(fallback)
    return float(mr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model_config', required=True,
                        help='e.g. unet / deeplabv3plus_resnet34 / segformer / baseline/transunet / research/modular_unet_pmsr_lite')
    parser.add_argument('--dataset', required=True,
                        help='e.g. beijing_1024_cart / beijing_1024_polar')
    parser.add_argument('--dataset_short', required=True,
                        help='e.g. beijing / tibet / zhengzhou')
    parser.add_argument('--split', default='test')
    parser.add_argument('--model_name', required=True,
                        help='e.g. DeepLabv3+_resnet34_cart')
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--max_radius', type=float, default=362)
    args = parser.parse_args()

    is_polar = 'polar' in args.dataset

    out_dir = Path(args.output_root) / args.model_name
    out_polar = out_dir / 'masks_polar' / args.dataset_short
    out_cart = out_dir / 'masks_cart' / args.dataset_short
    out_polar.mkdir(parents=True, exist_ok=True)
    out_cart.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {args.model_name} / {args.dataset_short} ({args.dataset}) ===')
    print(f'is_polar = {is_polar}')

    cfg = build_cfg(args.dataset, args.model_config, args.split, args.img_size, is_polar)
    datamodule = SegDataModule(cfg, eval_split=args.split)
    datamodule.setup(stage='test')
    loader = datamodule.test_dataloader()

    model = SegModule.load_from_checkpoint(args.ckpt, strict=False, cfg=cfg)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    center = ((args.img_size - 1) / 2, (args.img_size - 1) / 2)
    max_radius = resolve_max_radius(cfg, is_polar, args.max_radius)
    print(f'max_radius = {max_radius}')

    n_saved = 0
    for batch in tqdm(loader, desc=f'{args.model_name}/{args.dataset_short}'):
        images, masks, meta = unpack_batch(batch)
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            outputs = normalize_model_output(model(images))
            logits = outputs['seg_logits']
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for i in range(preds.shape[0]):
            case_id = meta['case_id'][i]
            pred = preds[i]

            if is_polar:
                Image.fromarray(pred).save(out_polar / f'{case_id}.png')
                cart = polar_mask_to_cart(
                    pred, out_h=args.img_size, out_w=args.img_size,
                    center=center, max_radius=max_radius, mode='linear',
                ).astype(np.uint8)
                Image.fromarray(cart).save(out_cart / f'{case_id}.png')
            else:
                Image.fromarray(pred).save(out_cart / f'{case_id}.png')
                polar = cart_to_polar_mask(
                    pred, center=center, max_radius=max_radius,
                    radius_bins=args.img_size, angle_bins=args.img_size,
                ).astype(np.uint8)
                Image.fromarray(polar).save(out_polar / f'{case_id}.png')

            n_saved += 1

    print(f'[INFO] Saved {n_saved} masks to {out_dir}')


if __name__ == '__main__':
    main()