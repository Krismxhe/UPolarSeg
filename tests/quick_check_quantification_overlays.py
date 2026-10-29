"""Create prediction/GT overlay examples without a checkpoint or external data.

Run from the project root:
    python quick_check_quantification_overlays.py --output_dir outputs/overlay_demo

The probabilities are deliberately uncertain near the PPA region. This example
shows the distinction between a hard boundary and soft radial occupation; it is
not a patient image or an experimental model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..src.evaluation.prediction_masks import PredictionMaskWriter
from ..src.evaluation.quantification_overlays import QuantificationOverlayWriter
from ..src.quantification.polar_readout import build_polar_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='outputs/overlay_demo')
    parser.add_argument('--dv_mode', choices=['fixed_vertical', 'max_center_chord'],
                        default='max_center_chord')
    parser.add_argument('--layout', choices=['combined', 'separate'], default='separate')
    parser.add_argument('--readouts', choices=['hard', 'soft', 'both'], default='both')
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    cfg = {
        'train': {'img_size': 512},
        'dataset': {'num_classes': 3, 'mask_mode': 'index'},
        'polar': {
            'enabled': True,
            'geometry': {'mode': 'linear', 'center': None, 'max_radius': None,
                         'angle_bins': 512, 'radius_bins': 512, 'pt_type': 'corner'},
        },
        'quantification': {'enabled': False, 'dv_mode': args.dv_mode,
                           'vertical_angle_deg': 90.0, 'denominator_floor_px': 1.0},
        'eval': {'save_predictions': 'both', 'prediction_cartesian_size': 'input',
                 'save_quantification_overlays': 'both',
                 'quantification_overlay_readouts': args.readouts,
                 'quantification_overlay_layout': args.layout,
                 'save_quantification_gt_overlays': 'both',
                 'quantification_overlay_legend': True},
    }
    readout = build_polar_readout(cfg)
    theta = torch.arange(readout.angle_bins, dtype=torch.float32)[:, None] * (2 * torch.pi / readout.angle_bins)
    radius = readout.radii[None, :]
    od_radius = 88 + 20 * torch.cos(2 * (theta - np.deg2rad(35))) + 12 * torch.cos(theta - np.deg2rad(10))
    ppa_width = 5 + 45 * torch.exp(4 * (torch.cos(theta - np.deg2rad(125)) - 1))
    p_od = torch.sigmoid((od_radius - radius) / 2.5)
    p_ppa = 0.68 * (1 - p_od) * torch.sigmoid((od_radius + ppa_width - radius) / 3.0)
    probabilities = torch.stack([1 - p_od - p_ppa, p_od, p_ppa], dim=0).unsqueeze(0)
    predictions = probabilities.argmax(dim=1)
    # A separate analytic target is used for the GT overlay. It is not obtained
    # by copying the prediction or converting its probabilities to one-hot.
    groundtruth = torch.where(radius <= od_radius, 1,
                              torch.where(radius <= od_radius + ppa_width, 2, 0)).long().unsqueeze(0)
    with torch.inference_mode():
        hard = readout.measure_mask(predictions)
        soft = readout.read_probabilities(probabilities)
        gt_hard = readout.measure_mask(groundtruth)
    assert bool(hard['reference_valid'][0]), 'The synthetic hard prediction should be readable.'
    mask = predictions[0].numpy()
    case_id = 'synthetic_example'
    mask_paths = PredictionMaskWriter(cfg, output).save(mask, case_id)
    overlay_paths = QuantificationOverlayWriter(cfg, output).save(
        mask, case_id, readout=readout,
        hard={key: value[0] for key, value in hard.items()},
        soft={key: value[0] for key, value in soft.items()},
        groundtruth_mask=groundtruth[0].numpy(),
        groundtruth_hard={key: value[0] for key, value in gt_hard.items()},
        prediction_source='seg_logits',
    )
    metadata = json.loads(Path(overlay_paths['overlay_metadata_path']).read_text())
    summary = {
        'synthetic': True, 'dv_mode': args.dv_mode, 'layout': args.layout,
        'hard': {name: float(hard[name][0]) for name in ('pmw', 'dv', 'pvdr')},
        'soft': {name: float(soft[name][0]) for name in ('pmw', 'dv', 'pvdr')},
        **mask_paths, **overlay_paths,
    }
    for kind in ('hard', 'soft'):
        if kind in metadata['readouts']:
            summary[kind]['dh'] = metadata['readouts'][kind]['dh']['value']
    print(json.dumps(summary, indent=2))

    # An optional side-by-side figure is for explanation only. The individual
    # output PNGs above keep the same dimensions as their corresponding masks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    variants = []
    if args.layout == 'combined':
        variants.append(('Prediction: ' + args.readouts, 'overlay_polar_path', 'overlay_cartesian_path'))
    else:
        for kind in ('hard', 'soft'):
            if args.readouts in (kind, 'both'):
                variants.append(('Prediction: ' + kind, f'overlay_polar_{kind}_path',
                                 f'overlay_cartesian_{kind}_path'))
    variants.append(('Ground truth: hard', 'overlay_gt_polar_path', 'overlay_gt_cartesian_path'))
    fig, axes = plt.subplots(2, len(variants), figsize=(5.2 * len(variants), 10.8),
                             facecolor='#f6f8fc', squeeze=False)
    for column, (title, polar_key, cart_key) in enumerate(variants):
        for row, key in enumerate((polar_key, cart_key)):
            axis = axes[row, column]
            axis.imshow(np.asarray(Image.open(overlay_paths[key])))
            axis.set_title(('Polar | ' if row == 0 else 'Cartesian | ') + title,
                           fontsize=12, fontweight='bold', pad=9)
            axis.axis('off')
    fig.suptitle('PMW / Dv / Dh on prediction and ground-truth masks',
                 fontsize=16, fontweight='bold', y=0.975)
    fig.text(0.5, 0.035,
             'Synthetic probabilities and a separate analytic ground truth; not patient data.\n'
             'Soft lines show equivalent occupation; Dh is perpendicular to the selected Dv.\n'
             f'Dv mode: {args.dv_mode}; values stay in resized Cartesian ROI pixels.',
             ha='center', va='center', fontsize=10, color='#344054')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.09, top=0.92, wspace=0.04, hspace=0.12)
    preview = output / 'overlay_preview.png'
    fig.savefig(preview, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Preview: {preview}')


if __name__ == '__main__':
    main()
