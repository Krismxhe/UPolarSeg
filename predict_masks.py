"""Export class-index predictions from a checkpoint on one dataset split.

--save_masks selects none, polar, cartesian, or both (default: both).
--save_overlays independently selects PMW/Dv/Dh visualizations (default: none).
--overlay_layout separates hard/soft predictions or combines them in one image.
--save_gt_overlays independently selects hard ground-truth overlays (default: none).
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
from src.evaluation.quantification_overlays import QuantificationOverlayWriter, normalize_overlay_mode
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
    parser.add_argument('--save_overlays', choices=['none', 'polar', 'cartesian', 'both'], default=None,
                        help='Quantification overlay spaces; default follows config (none)')
    parser.add_argument('--save_gt_overlays', choices=['none', 'polar', 'cartesian', 'both'], default=None,
                        help='Hard ground-truth overlay spaces; default follows config (none)')
    parser.add_argument('--overlay_layout', choices=['combined', 'separate'], default=None,
                        help='Prediction hard/soft image layout; default follows config (combined)')
    parser.add_argument('--overlay_readouts', choices=['hard', 'soft', 'both'], default=None,
                        help='Readouts to draw; default follows config (both)')
    parser.add_argument('--overlay_source', choices=['seg_logits', 'quantification'], default=None,
                        help='Prediction source to measure and colour; default follows config (seg_logits)')
    parser.add_argument('--input_space', choices=['auto', 'polar', 'cartesian'], default='auto',
                        help='auto retains the legacy inference from the dataset config name')
    parser.add_argument('--max_radius', type=float, default=None,
                        help='Optional override applied to both input transforms and export geometry')
    parser.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                        help='Repeat for training-config overrides, e.g. polar.geometry.mode=log')
    return parser


def main():
    args = build_parser().parse_args()

    is_polar = ('polar' in args.dataset.lower()) if args.input_space == 'auto' else args.input_space == 'polar'
    cfg = build_cfg(
        args.dataset, args.model_config, args.split, args.img_size, is_polar, args.override,
    )
    if args.max_radius is not None:
        OmegaConf.update(cfg, 'polar.geometry.max_radius', args.max_radius, force_add=True)
    OmegaConf.update(cfg, 'eval.save_predictions', args.save_masks)
    OmegaConf.update(cfg, 'eval.prediction_cartesian_size', args.cartesian_size)
    for key, value in (
        ('save_quantification_overlays', args.save_overlays),
        ('save_quantification_gt_overlays', args.save_gt_overlays),
        ('quantification_overlay_layout', args.overlay_layout),
        ('quantification_overlay_readouts', args.overlay_readouts),
        ('quantification_overlay_source', args.overlay_source),
    ):
        if value is not None:
            OmegaConf.update(cfg, f'eval.{key}', value, force_add=True)
    overlay_mode = normalize_overlay_mode(cfg.eval.get('save_quantification_overlays', 'none'))
    gt_overlay_mode = normalize_overlay_mode(cfg.eval.get('save_quantification_gt_overlays', 'none'))
    if args.save_masks == 'none' and overlay_mode == 'none' and gt_overlay_mode == 'none':
        print('[INFO] Mask, prediction-overlay and GT-overlay export disabled; no model or dataset loaded.')
        return

    out_dir = (Path(args.output_root) / args.model_name).resolve()
    writer = PredictionMaskWriter(
        cfg, out_dir,
        polar_dir=out_dir / 'masks_polar' / args.dataset_short,
        cartesian_dir=out_dir / 'masks_cart' / args.dataset_short,
    )
    overlay_writer = None
    if overlay_mode != 'none' or gt_overlay_mode != 'none':
        overlay_writer = QuantificationOverlayWriter(
            cfg, out_dir, mode=overlay_mode,
            polar_dir=out_dir / 'overlays_polar' / args.dataset_short,
            cartesian_dir=out_dir / 'overlays_cartesian' / args.dataset_short,
            metadata_dir=out_dir / 'overlay_metadata' / args.dataset_short,
            gt_polar_dir=out_dir / 'overlays_gt_polar' / args.dataset_short,
            gt_cartesian_dir=out_dir / 'overlays_gt_cartesian' / args.dataset_short,
        )
    print(f'\n=== {args.model_name} / {args.dataset_short} ({args.dataset}) ===')
    print(f'input_is_polar = {writer.input_is_polar}; save_masks = {writer.mode}; '
          f'save_overlays = {overlay_mode}; save_gt_overlays = {gt_overlay_mode}')

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
            images, masks, meta = unpack_batch(batch)
            overlay_batch = None
            if overlay_mode == 'none':
                outputs = normalize_model_output(model(images.to(device)))
                quant_logits, prediction_source = outputs['seg_logits'], 'seg_logits'
            else:
                outputs, quant_logits, prediction_source = model._evaluation_outputs(images.to(device))
            if overlay_writer is not None:
                overlay_batch = model._quantification_overlay_batch(
                    outputs['seg_logits'], quant_logits, prediction_source, target_masks=masks,
                )
            predictions = predict_from_logits(
                outputs['seg_logits'], is_binary=(int(cfg.dataset.num_classes) == 1),
                threshold=threshold,
            ).cpu().numpy()
            for index, prediction in enumerate(predictions):
                original_size = None
                mask_needs_original = (
                    writer.mode in {'cartesian', 'both'} and writer.cartesian_size == 'original'
                )
                if mask_needs_original or (overlay_writer is not None and overlay_writer.needs_original_size):
                    original_size = original_size_from_metadata(meta, index)
                    if original_size is None and meta.get('image_path'):
                        from PIL import Image
                        with Image.open(meta['image_path'][index]) as image:
                            original_size = image.height, image.width
                case_id = meta['case_id'][index]
                writer.save(prediction, case_id, original_size=original_size)
                if overlay_writer is not None:
                    complete = model._odppa_metadata.get(str(case_id), {}).get('reference_complete')
                    if complete in (None, ''):
                        complete = model._meta_get(meta, 'reference_complete', index, True)
                    model._save_quantification_overlay(
                        overlay_writer, overlay_batch, index, case_id, original_size=original_size,
                        reference_complete=complete,
                    )
                n_saved += 1

    print(f'[INFO] Exported predictions for {n_saved} cases.')
    if writer.mode in {'polar', 'both'}:
        print(f'  polar     -> {writer.polar_dir}')
    if writer.mode in {'cartesian', 'both'}:
        print(f'  cartesian -> {writer.cartesian_dir}')
    if overlay_writer is not None:
        for space, directory in (('polar', overlay_writer.polar_dir), ('cartesian', overlay_writer.cartesian_dir)):
            if overlay_mode not in {space, 'both'}:
                continue
            if overlay_writer.layout == 'separate':
                for kind in ('hard', 'soft'):
                    if overlay_writer.readouts in {kind, 'both'}:
                        print(f'  overlay {space} {kind} -> {directory / kind}')
            else:
                print(f'  overlay {space} -> {directory}')
        if gt_overlay_mode in {'polar', 'both'}:
            print(f'  GT overlay polar     -> {overlay_writer.gt_polar_dir}')
        if gt_overlay_mode in {'cartesian', 'both'}:
            print(f'  GT overlay cartesian -> {overlay_writer.gt_cartesian_dir}')
        print(f'  overlay metadata  -> {overlay_writer.metadata_dir}')


if __name__ == '__main__':
    main()
