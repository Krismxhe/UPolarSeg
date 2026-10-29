"""Render fundus-overlay exports on a synthetic RGB alignment fixture.

This example needs no checkpoint or patient image. Its colour background,
prediction, and reference are deterministic synthetic data, not clinical results.
Run from the project root with the project dependencies installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..src.evaluation.prediction_masks import _warp_mask
from ..src.evaluation.quantification_overlays import QuantificationOverlayWriter
from ..src.quantification.polar_readout import build_polar_readout


def _single(values):
    return {key: value[0] for key, value in values.items()}


def _rgb_fixture(reference_cartesian):
    """A textured RGB background with visible reference regions for alignment."""
    height, width = reference_cartesian.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx, yy = xx / width, yy / height
    texture = 5 * np.sin(xx * 47) * np.cos(yy * 39)
    illumination = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.34)
    rgb = np.stack((34 + 88 * illumination + texture,
                    18 + 42 * illumination + texture,
                    23 + 24 * illumination + texture), axis=-1)
    for class_id, color in ((1, (214, 176, 110)), (2, (171, 151, 119))):
        selected = reference_cartesian == class_id
        rgb[selected] = np.asarray(color) + texture[selected, None]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/fundus_overlay_demo')
    parser.add_argument('--dv_mode', choices=['fixed_vertical', 'max_center_chord'],
                        default='max_center_chord')
    parser.add_argument('--alpha', type=float, default=0.0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = {
        'train': {'img_size': 512},
        'dataset': {'num_classes': 3, 'mask_mode': 'index'},
        'polar': {'enabled': True,
                  'geometry': {'mode': 'linear', 'center': None, 'max_radius': None,
                               'angle_bins': 512, 'radius_bins': 512, 'pt_type': 'corner'}},
        'quantification': {'enabled': False, 'dv_mode': args.dv_mode,
                           'vertical_angle_deg': 90.0, 'denominator_floor_px': 1.0},
        'eval': {'save_predictions': 'none',
                 'save_quantification_overlays': 'none',
                 'save_quantification_gt_overlays': 'none',
                 # Deliberately select soft for the old mask overlays. Fundus
                 # exports independently request and draw only hard geometry.
                 'quantification_overlay_readouts': 'soft',
                 'quantification_overlay_layout': 'separate',
                 'quantification_overlay_source': 'seg_logits',
                 'quantification_overlay_legend': True,
                 'prediction_cartesian_size': 'original',
                 'save_fundus_overlays': 'both',
                 'fundus_overlay_alpha': args.alpha},
    }
    readout = build_polar_readout(cfg)
    theta = torch.arange(readout.angle_bins, dtype=torch.float32)[:, None] * (
        2 * torch.pi / readout.angle_bins
    )
    radius = readout.radii[None, :]
    reference_radius = 82 + 19 * torch.cos(2 * (theta - np.deg2rad(32)))
    reference_width = 4 + 40 * torch.exp(4 * (torch.cos(theta - np.deg2rad(125)) - 1))
    prediction_radius = 84 + 17 * torch.cos(2 * (theta - np.deg2rad(39)))
    prediction_width = 3 + 36 * torch.exp(4 * (torch.cos(theta - np.deg2rad(132)) - 1))

    def labels(od_radius, ppa_width):
        return torch.where(radius <= od_radius, 1,
                           torch.where(radius <= od_radius + ppa_width, 2, 0)).long().unsqueeze(0)

    prediction = labels(prediction_radius, prediction_width)
    reference = labels(reference_radius, reference_width)
    with torch.inference_mode():
        hard = readout.measure_mask(prediction)
        gt_hard = readout.measure_mask(reference)
    assert bool(hard['reference_valid'][0]) and bool(gt_hard['reference_valid'][0])
    reference_cartesian = _warp_mask(reference[0].numpy(), cfg, inverse=True)
    source = cv2.resize(_rgb_fixture(reference_cartesian), (768, 768),
                        interpolation=cv2.INTER_LINEAR)
    source_path = output / 'synthetic_rgb_source.png'
    Image.fromarray(source).save(source_path)
    paths = QuantificationOverlayWriter(cfg, output).save(
        prediction[0].numpy(), 'synthetic_example', readout=readout,
        hard=_single(hard), soft=None, groundtruth_mask=reference[0].numpy(),
        groundtruth_hard=_single(gt_hard), original_size=source.shape[:2],
        image_path=str(source_path), prediction_source='seg_logits',
    )
    summary = {'synthetic': True, 'dv_mode': args.dv_mode, 'alpha': args.alpha,
               'source_image': str(source_path), **paths}
    (output / 'demo_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 6.0), facecolor='#f6f8fc')
    panels = [('Source RGB fixture', source_path),
              ('Reference contours + hard measurements', Path(paths['overlay_gt_fundus_path'])),
              ('Prediction contours + hard measurements', Path(paths['overlay_fundus_path']))]
    for axis, (title, path) in zip(axes, panels, strict=True):
        with Image.open(path) as image:
            axis.imshow(np.asarray(image))
        axis.set_title(title, fontsize=11, fontweight='bold', pad=10)
        axis.axis('off')
    fig.suptitle('Fundus overlay rendering | ' + args.dv_mode,
                 fontsize=16, fontweight='bold', y=0.97)
    fig.text(0.5, 0.035,
             'Synthetic RGB alignment fixture and independent analytic masks; no patient data or model results.\n'
             '512-pixel input geometry is mapped to a 768-pixel source image; numerical lengths remain input ROI pixels.',
             ha='center', fontsize=10, color='#344054')
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.12, top=0.88, wspace=0.025)
    fig.savefig(output / 'fundus_preview.png', dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
