# Experiment configs & ablation protocol

Each file here is a Hydra **experiment** (`# @package _global_`). It pins a full,
reproducible configuration by overriding the default groups. Run with the `+`
(append) syntax:

```bash
python train.py +experiment=smp_unet_resnet34
python train.py +experiment=smp_unetplusplus_resnet34
python train.py +experiment=smp_deeplabv3plus_resnet50
python train.py +experiment=modular_unet_identity
python train.py +experiment=transunet_r50_vit_b16
```

Any field can still be overridden on top of an experiment, e.g.:

```bash
python train.py +experiment=smp_unet_resnet34 train.epochs=200 train.seed=7
```

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

## Reproducibility / seed sweeps

```bash
python train.py --multirun +experiment=smp_unet_resnet34 train.seed=42,43,44
```

## Skip-module ablation (ModularUNet)

Only the `identity` skip module is implemented today, so a runnable ablation is a
seed sweep of identity:

```bash
python train.py --multirun +experiment=modular_unet_identity train.seed=42,43,44
```

> **TODO (not yet runnable):** once additional skip modules (`scse`, `cbam`, …)
> are registered in `src/models/modular_unet/skip_modules.py::SKIP_MODULES`, the
> intended ablation is:
>
> ```bash
> # PLANNED — scse/cbam not implemented yet, do not run as-is
> # python train.py --multirun +experiment=modular_unet_identity \
> #     model.skip.name=identity,scse,cbam train.seed=42,43,44
> ```

## Dataset paths

These experiments use the default `dataset=multiclass` config. Its `root` is
defined in `configs/dataset/multiclass.yaml` — no machine-specific path is
hard-coded here. Point a run at your own data with `dataset=<your_dataset>` or
`dataset.root=<path>`.
