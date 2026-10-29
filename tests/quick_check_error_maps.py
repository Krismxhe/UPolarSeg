"""Render class-wise errors on synthetic RGB data, without a checkpoint.

The native Cartesian GT is generated independently at 768 x 768. Model-grid
predictions are deterministic synthetic masks, not learned model results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..src.evaluation.error_maps import ErrorMapWriter
from ..src.evaluation.prediction_masks import _warp_mask


def _labels(shape, *, prediction=False):
    height, width = shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    # Define both native and model-grid shapes in the same 512-grid coordinates.
    xx = (xx + 0.5) * 512 / width - 0.5
    yy = (yy + 0.5) * 512 / height - 0.5
    cx, cy = (259, 251) if prediction else (251, 257)
    angle = np.deg2rad(24)
    dx, dy = xx - cx, yy - cy
    u, v = dx * np.cos(angle) + dy * np.sin(angle), -dx * np.sin(angle) + dy * np.cos(angle)
    od = (u / 86) ** 2 + (v / (64 if prediction else 74)) ** 2 <= 1
    outer = ((xx - (276 if prediction else 266)) / (113 if prediction else 119)) ** 2
    outer += ((yy - (274 if prediction else 269)) / (104 if prediction else 98)) ** 2
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(outer <= 1) & ~od] = 2
    mask[od] = 1
    return mask


def _rgb_fixture(mask):
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx, yy = xx / width, yy / height
    illumination = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.3)
    texture = 4 * np.sin(xx * 47) * np.cos(yy * 31)
    rgb = np.stack((34 + 87 * illumination + texture,
                    20 + 42 * illumination + texture,
                    22 + 25 * illumination + texture), axis=-1)
    for class_id, color in ((1, (216, 178, 112)), (2, (161, 140, 111))):
        selected = mask == class_id
        rgb[selected] = np.asarray(color) + texture[selected, None]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/error_map_demo')
    parser.add_argument('--input_space', choices=['polar', 'cartesian'], default='polar')
    parser.add_argument('--alpha', type=float, default=0.65)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = {
        'train': {'img_size': 512},
        'dataset': {'num_classes': 3, 'mask_mode': 'index'},
        'polar': {'enabled': args.input_space == 'polar', 'geometry': {
            'mode': 'linear', 'center': None, 'max_radius': None,
            'angle_bins': 512, 'radius_bins': 512, 'pt_type': 'corner',
        }},
        'quantification': {'enabled': False, 'od_class': 1, 'ppa_class': 2},
        'eval': {
            'save_error_maps': 'both', 'save_error_fundus': True,
            'error_map_classes': 'both', 'error_map_source': 'seg_logits',
            'error_map_alpha': args.alpha, 'error_map_legend': True,
            'prediction_cartesian_size': 'original',
        },
    }
    gt_native = _labels((768, 768))
    pred_cartesian = _labels((512, 512), prediction=True)
    gt_input = cv2.resize(gt_native, (512, 512), interpolation=cv2.INTER_NEAREST)
    prediction, target = pred_cartesian, gt_input
    if cfg['polar']['enabled']:
        prediction = _warp_mask(prediction, cfg, inverse=False)
        target = _warp_mask(target, cfg, inverse=False)
    rgb = _rgb_fixture(gt_native)
    image_path, mask_path = output / 'synthetic_rgb_source.png', output / 'native_gt.png'
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(gt_native).save(mask_path)
    Image.fromarray(prediction).save(output / 'prediction_model_grid.png')
    Image.fromarray(target).save(output / 'gt_model_grid.png')
    writer = ErrorMapWriter(cfg, output)
    paths = writer.save(
        prediction, 'synthetic_example', groundtruth_mask=target,
        mask_path=str(mask_path), image_path=str(image_path),
        original_size=gt_native.shape, reference_complete=True,
        prediction_source='seg_logits',
    )
    summary = {'synthetic': True, 'input_space': args.input_space,
               'native_gt_shape': list(gt_native.shape), 'alpha': args.alpha, **paths}
    (output / 'demo_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    restored = _warp_mask(prediction, cfg, inverse=True) if cfg['polar']['enabled'] else prediction
    restored = cv2.resize(restored, (768, 768), interpolation=cv2.INTER_NEAREST)
    palette = np.array([[13, 20, 33], [96, 165, 250], [232, 121, 249]], dtype=np.uint8)
    def read_rgb(key):
        with Image.open(paths[key]) as im:
            return np.array(im.convert('RGB'))
    panels = [
        ('Synthetic source RGB', rgb),
        ('OD errors on RGB', read_rgb('error_od_fundus_path')),
        ('PPA errors on RGB', read_rgb('error_ppa_fundus_path')),
        ('Native Cartesian reference', palette[gt_native]),
        ('Restored prediction', palette[restored]),
        ('PPA error map: red FP / blue FN', read_rgb('error_ppa_cartesian_path')),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 9.7), facecolor='#f6f8fc')
    for axis, (title, values) in zip(axes.flat, panels, strict=True):
        axis.imshow(values)
        axis.set_title(title, fontsize=11, fontweight='bold', pad=7)
        axis.axis('off')
    fig.suptitle('Class-wise error maps | ' + args.input_space + ' model grid',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.025,
             'Synthetic images and predictions; no patient data or model-performance results.\n'
             '512-grid predictions are restored to the 768-grid native GT before Cartesian errors are computed.',
             ha='center', fontsize=10, color='#344054')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.09, wspace=0.04, hspace=0.13)
    fig.savefig(output / 'error_map_preview.png', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
