"""Export class-index predictions from a checkpoint on one dataset split.

--save_masks selects none, polar, cartesian, or both (default: both).
The existing CLI directory layout is retained:
    {output_root}/{model_name}/masks_polar/{dataset_short}/{case_id}.png
    {output_root}/{model_name}/masks_cart/{dataset_short}/{case_id}.png

Cartesian output defaults to each original image's size. Pass
--cartesian_size input to retain the previous input-grid output dimensions.
Geometry comes from the resolved configuration, with optional Hydra overrides.
"""

import argparse
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets.seg_dataset import SegDataModule
from src.datasets.batch import original_size_from_metadata, unpack_batch
from src.evaluation.prediction_masks import PredictionMaskWriter
from src.metrics.functional import predict_from_logits
from src.models.seg_module import SegModule
from src.models.outputs import normalize_model_output


def build_cfg(dataset, model_config, split, img_size, is_polar, extra_overrides=None):
    overrides = [
        f'dataset={dataset}',
        f'model={model_config}',
        f'split={split}',
        f'train.img_size={img_size}',
        'hardware.devices=[0]',
        f'polar={"enabled" if is_polar else "disabled"}',
    ]
    overrides.extend(extra_overrides or [])
    with hydra.initialize(version_base=None, config_path='configs'):
        return hydra.compose(config_name='train', overrides=overrides)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model_config', required=True,
                        help='Model config used for training, e.g. unet or baseline/transunet')
    parser.add_argument('--dataset', required=True,
                        help='Dataset config, e.g. beijing_1024_cart or beijing_1024_polar')
    parser.add_argument('--dataset_short', required=True,
                        help='Dataset subdirectory name, e.g. beijing')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_masks', '--save_predictions', dest='save_masks',
                        choices=['none', 'polar', 'cartesian', 'both'], default='both')
    parser.add_argument('--cartesian_size', choices=['original', 'input'], default='original')
    parser.add_argument('--input_space', choices=['auto', 'polar', 'cartesian'], default='auto',
                        help='auto retains the legacy inference from the dataset config name')
    parser.add_argument('--max_radius', type=float, default=None,
                        help='Optional override applied to both input transforms and export geometry')
    parser.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                        help='Repeat for training-config overrides, e.g. polar.geometry.mode=log')
    return parser


def main():
    args = build_parser().parse_args()
    if args.save_masks == 'none':
        print('[INFO] Mask export disabled; no model or dataset loaded and no masks written.')
        return

    is_polar = ('polar' in args.dataset.lower()) if args.input_space == 'auto' else args.input_space == 'polar'
    cfg = build_cfg(
        args.dataset, args.model_config, args.split, args.img_size, is_polar, args.override,
    )
    if args.max_radius is not None:
        OmegaConf.update(cfg, 'polar.geometry.max_radius', args.max_radius, force_add=True)
    OmegaConf.update(cfg, 'eval.save_predictions', args.save_masks)
    OmegaConf.update(cfg, 'eval.prediction_cartesian_size', args.cartesian_size)

    out_dir = (Path(args.output_root) / args.model_name).resolve()
    writer = PredictionMaskWriter(
        cfg, out_dir,
        polar_dir=out_dir / 'masks_polar' / args.dataset_short,
        cartesian_dir=out_dir / 'masks_cart' / args.dataset_short,
    )
    print(f'\n=== {args.model_name} / {args.dataset_short} ({args.dataset}) ===')
    print(f'input_is_polar = {writer.input_is_polar}; save_masks = {writer.mode}')

    datamodule = SegDataModule(cfg, eval_split=str(cfg.split))
    datamodule.setup(stage='test')
    loader = datamodule.test_dataloader()

    model = SegModule.load_from_checkpoint(args.ckpt, strict=False, cfg=cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    threshold = float(cfg.eval.get('threshold', 0.5))
    n_saved = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f'{args.model_name}/{args.dataset_short}'):
            images, _masks, meta = unpack_batch(batch)
            outputs = normalize_model_output(model(images.to(device)))
            predictions = predict_from_logits(
                outputs['seg_logits'], is_binary=(int(cfg.dataset.num_classes) == 1),
                threshold=threshold,
            ).cpu().numpy()
            for index, prediction in enumerate(predictions):
                original_size = None
                if writer.mode in {'cartesian', 'both'} and writer.cartesian_size == 'original':
                    original_size = original_size_from_metadata(meta, index)
                writer.save(prediction, meta['case_id'][index], original_size=original_size)
                n_saved += 1

    print(f'[INFO] Saved predictions for {n_saved} cases.')
    if writer.mode in {'polar', 'both'}:
        print(f'  polar     -> {writer.polar_dir}')
    if writer.mode in {'cartesian', 'both'}:
        print(f'  cartesian -> {writer.cartesian_dir}')


if __name__ == '__main__':
    main()
