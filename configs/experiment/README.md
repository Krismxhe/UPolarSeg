# Experiment configs & ablation protocol

Each file here is a Hydra **experiment** (`# @package _global_`). It pins a full,
reproducible configuration by overriding the default groups. Experiments are
organized into three layers — but they all share the **same** dataset,
datamodule, augmentation, loss, optimizer, scheduler, trainer, evaluator, and
CSV schema. Only the `model` (and, for an ablation, the one factor under study)
should differ. That shared pipeline is what makes baseline-vs-research
comparisons fair.

```text
configs/experiment/
├── baselines/    established baselines (SMP Unet/Unet++/DeepLabV3+, TransUNet)
├── research/     user-owned research methods (ModularUNet, PolarSeg, …)
├── ablations/    ablations OF a research method (skip modules, boundary loss, …)
└── <flat>.yaml   DEPRECATED flat aliases that forward to the relocated configs
```

- **baselines/** — `provider=smp` and `provider=baseline` reference points.
- **research/** — `provider=research` methods (your own models).
- **ablations/** — controlled single-factor studies of a research method.

## Runnable experiments

```bash
python train.py +experiment=baselines/smp_unet_resnet34
python train.py +experiment=baselines/smp_unetplusplus_resnet34
python train.py +experiment=baselines/smp_deeplabv3plus_resnet50
python train.py +experiment=baselines/transunet_r50_vit_b16 train.img_size=224
python train.py +experiment=research/modular_unet_identity
```

Any field can still be overridden on top of an experiment, e.g.:

```bash
python train.py +experiment=baselines/smp_unet_resnet34 train.epochs=200 train.seed=7
```

## Templates / TODO (NOT runnable as-is)

These are **commented examples** documenting intended experiments whose
underlying model/feature is not implemented or wired yet. Do **not** run them —
they carry no overrides and would silently fall back to the default config.

| File | Why it's a template |
|---|---|
| `research/polarseg_v1.yaml` | PolarSeg model not implemented (no `configs/model/research/polarseg.yaml`). |
| `ablations/skip_modules.yaml` | Only the `identity` skip ships; `scse`/`cbam` not registered. |
| `ablations/boundary_loss.yaml` | No boundary head / boundary loss wired into training (boundary metrics are eval-only). |
| `ablations/clinical_metrics.yaml` | Clinical metrics are deterministic & eval-only (pixel units); no training term. |
| `ablations/output_heads.yaml` | Multi-output heads + `MultiTaskLoss` are skeletons, not wired into training. |

## Migration from the old flat paths

Experiment configs used to live flat (`+experiment=smp_unet_resnet34`). They now
live under `baselines/` and `research/`. The old flat names are **kept as
deprecated wrapper configs** that forward to the new location, so existing
commands keep working:

```bash
# old (still works via a deprecated forwarding wrapper)
python train.py +experiment=smp_unet_resnet34
# new (preferred)
python train.py +experiment=baselines/smp_unet_resnet34
```

| Old flat alias | New canonical path |
|---|---|
| `smp_unet_resnet34` | `baselines/smp_unet_resnet34` |
| `smp_unetplusplus_resnet34` | `baselines/smp_unetplusplus_resnet34` |
| `smp_deeplabv3plus_resnet50` | `baselines/smp_deeplabv3plus_resnet50` |
| `transunet_r50_vit_b16` | `baselines/transunet_r50_vit_b16` |
| `modular_unet_identity` | `research/modular_unet_identity` |

Prefer the new paths in new scripts; the flat wrappers may be removed later.

## What gets tracked

Every run writes a fully-resolved `config.yaml` next to its evaluation CSVs
(`outputs/<logging.name>/eval/<split>/config.yaml`), which captures all of:

| Tracked item | Where it lives in the config |
|---|---|
| model provider / name | `model.provider`, `model.name` |
| encoder / backbone | `model.encoder` (TransUNet: `model.vit_name`) |
| pretrained weights | `model.encoder_weights`, `model.pretrained_path` |
| loss | `loss.name` (+ weights) |
| optimizer / scheduler | `optimizer.*`, `scheduler.*` |
| train image size | `train.img_size` |
| augmentation | `augmentation` group |
| dataset | `dataset` group (name / root / classes) |
| seed | `train.seed` |
| checkpoint path | `outputs/<logging.name>/.../checkpoints/best.ckpt` |
| evaluation split | `split` (passed to `evaluate.py`) |
| summary / per_class / per_case CSV | `outputs/<logging.name>/eval/<split>/*.csv` |

The `summary.csv` row also records `run_name, checkpoint, dataset, split, model,
encoder, num_cases, mean_dice, mean_iou, mean_precision, mean_recall`.

## Reproducibility / seed sweeps & comparisons

```bash
# seed sweep of one experiment
python train.py --multirun +experiment=baselines/smp_unet_resnet34 train.seed=42,43,44
python train.py --multirun +experiment=research/modular_unet_identity train.seed=42,43,44

# baseline-vs-research model sweep over the SAME shared pipeline
python train.py --multirun model=smp/unet,smp/deeplabv3plus,baseline/transunet train.seed=42,43,44
```

Because every experiment overrides only the `model` group (plus the shared
dataset/aug/loss/optimizer), all of the above are evaluated by the same
`evaluator` with the same metric/CSV schema — a fair comparison.

## Skip-module ablation (ModularUNet)

Only the `identity` skip module is implemented today, so the runnable ablation
is a seed sweep of identity (see `ablations/skip_modules.yaml` for the planned,
not-yet-runnable scse/cbam sweep):

```bash
python train.py --multirun +experiment=research/modular_unet_identity train.seed=42,43,44
```

## Dataset paths

These experiments use the default `dataset=multiclass` config. Its `root` is
defined in `configs/dataset/multiclass.yaml` — no machine-specific path is
hard-coded here. Point a run at your own data with `dataset=<your_dataset>` or
`dataset.root=<path>`.
