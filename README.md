# seg-baseline

A lightweight, config-driven research framework for **2D medical image
segmentation**. It pairs a fair, shared training/evaluation pipeline with a
clean separation between established baselines and your own research models, so
comparisons stay honest and new methods are easy to add.

Built on [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) (SMP),
[PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/).

## Key features

- **Config-driven** — swap model, backbone, loss, optimizer, and augmentation
  from the command line; no code changes.
- **Three model layers, one pipeline** — `smp` (SMP architectures), `baseline`
  (established non-SMP baselines such as TransUNet), and `research` (your own
  methods such as ModularUNet). Implementations are separated; dataset,
  augmentation, loss, optimizer, trainer, and evaluator are shared — so a
  baseline and a research model are compared under identical conditions.
- **Multi-class and binary** segmentation through a single code path.
- **Explicit evaluation CSVs** — `summary.csv`, `per_class.csv`, and
  `per_case.csv` written on every evaluation run, ready for analysis.
- **Unified output contract** — a model may return a logits tensor or a dict;
  the framework normalizes both, so multi-output models drop in without touching
  the training loop.
- **Reproducible experiments** — pinned Hydra experiment configs you run with
  `+experiment=<name>`, each saving its fully-resolved config alongside results.

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `segmentation-models-pytorch`,
`albumentations`, `hydra-core`, `torchmetrics` (plus `pytest` for tests).

## Quickstart

```bash
# train a UNet/ResNet34 baseline (defaults: multiclass, dice_ce)
python train.py model=smp/unet

# evaluate a checkpoint and write CSVs
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt \
    split=test eval.save_csv=true

# run on a single image
python predict.py --img path/to/image.png \
    --checkpoint outputs/<name>/checkpoints/best.ckpt --out result.png
```

## Models — the three provider layers

A model config carries a `provider` field, and `build_model` routes on it. The
three layers separate *who owns the implementation*; everything downstream is
shared (see [Shared pipeline](#shared-pipeline)).

### `provider=smp` — SMP baselines

Architectures built directly through `segmentation_models_pytorch`
(`Unet`, `UnetPlusPlus`, `DeepLabV3Plus`, `FPN`, `MAnet`, …).

```bash
python train.py model=smp/unet
python train.py model=smp/deeplabv3plus model.encoder=resnet50
python train.py model=smp/unetplusplus model.encoder=resnet50

# legacy configs without a provider field still resolve to smp
python train.py model=unet
```

### `provider=baseline` — established non-SMP baselines

Baselines SMP does not provide. **TransUNet** ships today — a hybrid model where
an SMP ResNet50 backbone feeds its stride-16 feature map to a ViT encoder, then a
cascaded decoder with CNN skip connections restores full resolution. (UNETR /
SwinUNETR are on the roadmap.)

```bash
python train.py model=baseline/transunet train.img_size=224
```

> `img_size` must be divisible by `patch_size` (16). The ResNet backbone is
> ImageNet-initialised via `model.encoder_weights`; the transformer trains from
> scratch unless `model.pretrained_path` points to a compatible torch
> `state_dict` of this model (see [Limitations](#limitations)).

### `provider=research` — your own methods

User-owned research implementations. **ModularUNet** ships today — a controllable
UNet (SMP encoder + a custom decoder) for skip-connection research; it currently
provides the `identity` skip as an ablation control.

```bash
python train.py model=research/modular_unet_identity
python train.py model=research/modular_unet model.skip.name=identity
```

> `img_size` must be divisible by 32 (resnet-style 5-stage encoder).

### `provider=custom` — deprecated alias

Kept only so old configs and commands keep working: it maps `custom/transunet →
baseline/transunet` and `custom/modular_unet → research/modular_unet`. Prefer
`baseline` / `research` in new work — see [Migration](#migration).

## Shared pipeline

Every provider shares one pipeline, so the only thing that differs between a
baseline run and a research run is the model (unless an experiment deliberately
studies another factor):

- dataset, datamodule, and augmentation
- the batch contract and the model **output contract**
- the loss factory, optimizer, and scheduler
- the Lightning trainer and the LightningModule
- the evaluator and its `summary` / `per_class` / `per_case` CSV schema

The training module imports no concrete model class, and neither the evaluator
nor the loss branches on provider. Same data, same metrics, same CSVs — that is
what keeps baseline-vs-research comparisons fair.

## Migration

Nothing breaks: old configs and commands still run via the deprecated `custom`
alias and flat experiment wrappers. Preferred new forms:

| Old | New |
|---|---|
| `model=custom/transunet` | `model=baseline/transunet` |
| `model=custom/modular_unet` | `model=research/modular_unet_identity` (or `model=research/modular_unet`) |
| `+experiment=transunet_r50_vit_b16` | `+experiment=baselines/transunet_r50_vit_b16` |
| `+experiment=modular_unet_identity` | `+experiment=research/modular_unet_identity` |

A standalone guide lives in [`docs/MIGRATION.md`](docs/MIGRATION.md).

## Datasets

```
your-dataset/
├── train/
│   ├── images/        ← RGB images (.png / .jpg)
│   └── <mask_dir>/    ← masks (.png, same stem as the image)
├── val/   …
└── test/  …
```

| `mask_mode` | pixel values | use case |
|---|---|---|
| `index`  | 0, 1, 2, … (class indices) | multi-class |
| `binary` | 0 / 255 → auto-converted to 0 / 1 | binary |

### Binary vs multi-class

| | binary | multi-class |
|---|---|---|
| `num_classes` | 1 | > 1 |
| model output | `B×1×H×W` | `B×C×H×W` |
| prediction | `sigmoid(logits) > eval.threshold` | `argmax(logits, dim=1)` |
| loss | `dice_bce` | `dice_ce` |
| mean metrics | the single foreground class | over `foreground_classes` (background excluded by default) |

### Add a dataset

Copy `configs/dataset/multiclass.yaml` (or `binaryclass.yaml`), edit, and run
`python train.py dataset=my_dataset`:

```yaml
name: my_dataset
root: /path/to/dataset
mask_dir: masks
mask_mode: index
num_classes: 3
class_names: [background, class_a, class_b]
foreground_classes: [1, 2]   # used for mean Dice/IoU (excludes background)
```

## Loss

The loss is selected by the dataset's task type:

| Loss | Task (`num_classes`) | Composition |
|---|---|---|
| `dice_ce` (default) | multi-class (> 1) | Dice + CrossEntropy |
| `dice_bce` | binary (== 1) | Dice + BCEWithLogits |

```bash
python train.py loss=dice_ce                      # default (multi-class)
python train.py dataset=binaryclass loss=dice_bce # binary
```

Losses are computed from logits — models never apply sigmoid/softmax. Using
`dice_ce` on a binary task (or vice versa) raises a clear error.

## Training

```bash
python train.py                       # UNet + ResNet34, multiclass, dice_ce
python train.py train.batch_size=4 train.epochs=200 train.img_size=640
python train.py optimizer=sgd optimizer.lr=1e-2 augmentation=heavy

# multi-GPU / mixed precision, and Hydra multirun sweeps
python train.py hardware.devices=4 hardware.strategy=ddp train.precision=16-mixed
python train.py --multirun model=smp/unet,smp/unetplusplus model.encoder=resnet34,resnet50
```

## Experiments & reproducibility

Pinned configurations live in `configs/experiment/`, grouped into `baselines/`,
`research/`, and `ablations/`. Run one with the append (`+`) syntax:

```bash
python train.py +experiment=baselines/smp_unet_resnet34
python train.py +experiment=baselines/smp_deeplabv3plus_resnet50
python train.py +experiment=baselines/transunet_r50_vit_b16 train.img_size=224
python train.py +experiment=research/modular_unet_identity

# seed sweep
python train.py --multirun +experiment=research/modular_unet_identity train.seed=42,43,44
```

Each experiment pins the model, dataset, augmentation, optimizer, loss,
`train.img_size`, `train.seed`, and a stable `logging.name`; the fully-resolved
config is saved next to the evaluation CSVs. See
[`configs/experiment/README.md`](configs/experiment/README.md) for the full list,
the flat-name compatibility wrappers, and the (not-yet-runnable) ablation
templates.

## Evaluation & CSV outputs

```bash
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=test eval.save_csv=true
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=val
```

Alongside the printed metric table, CSVs are written to
`outputs/<logging.name>/eval/<split>/`:

| file | one row per | key columns |
|---|---|---|
| `summary.csv` | evaluation run | `num_cases, mean_dice, mean_iou, mean_precision, mean_recall` (+ run/model/encoder) |
| `per_class.csv` | class | `class_id, class_name, dice, iou, precision, recall, support_pixels` |
| `per_case.csv` | case | `case_id, image_path, mask_path, dice_mean, iou_mean, precision_mean, recall_mean, pred_path` (+ per-class columns) |
| `config.yaml` | — | the fully-resolved config |

Relevant `eval.*` switches (`configs/train.yaml`):

```yaml
eval:
  save_csv: true
  threshold: 0.5            # binary prediction threshold
  include_background: false # include background in mean_* if true
  output_dir: null          # default: <save_dir>/<logging.name>/eval/<split>
  per_case: true
  per_class: true
```

### Metric conventions

- Aggregation is **macro**: per-case metrics are averaged across cases, and the
  summary averages over foreground classes (background excluded unless
  `include_background=true`).
- **Empty masks** are handled explicitly so metrics never become `NaN`:
  - pred empty **and** target empty → `dice = 1, iou = 1`
  - pred non-empty **and** target empty → `dice = 0, iou = 0`
  - pred empty **and** target non-empty → `dice = 0, iou = 0`
  - precision / recall follow the same rule.

## Optional eval outputs

Two extra, evaluation-only metric families can be enabled per run. Both are
deterministic and derived from the predicted segmentation — they do not require a
specialized model head.

```bash
# clinical morphology metrics → clinical_metrics.csv (one row per case)
python evaluate.py checkpoint=... split=test task.outputs.clinical.enabled=true

# boundary metrics → boundary_metrics.csv (boundary dice/iou per class + means)
python evaluate.py checkpoint=... split=test \
    task.outputs.boundary.enabled=true task.outputs.boundary.boundary_width_px=3
```

Clinical metrics are reported in **pixel units** (each row carries a `unit`
column). Both are off by default and never enter the training loss.

## Output contract

A model's `forward` may return either a logits tensor `B×C×H×W` or a dict that
contains at least `seg_logits`:

```python
logits = model(images)
# or
{
    "seg_logits": seg_logits,
    "boundary_logits": ...,   # optional
    "clinical": ...,          # optional
    "features": ...,          # optional
}
```

The framework normalizes both forms before computing loss and metrics, so a
multi-output model integrates without changing the training loop. A dict missing
`seg_logits` raises a clear error.

## Extending the framework

### Add a baseline model (`provider=baseline`)

1. Implement the `torch.nn.Module` under `src/models/baselines/<new_model>/`.
2. Register a builder in `src/models/baselines/registry.py`.
3. Add `configs/model/baseline/<new_model>.yaml` with `provider: baseline`,
   `name`, and `arch:` + `encoder:` labels (used for the run name).
4. Make `forward` honour the [output contract](#output-contract).
5. Add a forward-shape test (see `tests/test_smoke.py`).

### Add a research model (`provider=research`)

1. Implement the `torch.nn.Module` under `src/models/research/<your_method>/`
   (`ResearchModel` in `src/models/research/base.py` is an optional base).
2. Register a builder in `src/models/research/registry.py`.
3. Add `configs/model/research/<your_method>.yaml` with `provider: research`,
   `name`, and `arch:` + `encoder:` labels.
4. Make `forward` honour the [output contract](#output-contract); emit
   `boundary_logits` / `clinical` / `features` if your method produces them.
5. Add a forward-shape test, and don't modify the shared pipeline.

Registries are the only place model names are dispatched — `factory.py` and the
LightningModule stay untouched. Baseline and research code must not import each
other's internals.

### Add a skip module (ModularUNet)

```python
# src/models/research/modular_unet/skip_modules.py
class MySkip(SkipModule):
    def forward(self, skip, decoder_feature=None, level: int = 0):
        return skip  # must preserve spatial size
SKIP_MODULES["my_skip"] = MySkip
```

Select it with `model.skip.name=my_skip` and ablate against the identity skip:

```bash
python train.py --multirun model=research/modular_unet \
    model.skip.name=identity,my_skip train.seed=42,43,44
```

## Project structure

```
├── configs/
│   ├── train.yaml              ← main config (model/dataset/aug/optimizer/loss/task)
│   ├── model/{smp,baseline,research,custom}/
│   ├── dataset/  augmentation/  optimizer/  loss/  task/
│   └── experiment/{baselines,research,ablations}/
├── src/
│   ├── datasets/               ← dataset + dict batch contract
│   ├── models/
│   │   ├── factory.py          ← build_model (the provider router)
│   │   ├── smp_models.py  outputs.py  seg_module.py
│   │   ├── baselines/          ← registry + transunet/
│   │   └── research/           ← registry, base, modular_unet/
│   ├── losses/  metrics/  tasks/  transforms/  utils/
├── train.py   evaluate.py   predict.py
├── tests/test_smoke.py
└── requirements.txt
```

## Limitations

- **TransUNet pretrained weights** — only a partial torch `state_dict` load
  (`strict=False`) is supported; the official Google ViT/R50 `.npz` weights are
  not. The transformer trains from scratch unless given a compatible checkpoint.
- **Clinical metrics are pixel-unit only** — no physical (mm / mm²) conversion.
- **Boundary metrics are derived from the segmentation prediction** — there is no
  learned boundary head or boundary loss in training.
- **Multi-output training** — the clinical/boundary outputs are evaluation-only;
  joint multi-task training is not yet available.
- **Extra skip modules** (scse, cbam, …) are not implemented yet; only `identity`
  ships.
- **CSV export is single-process** — run evaluation on a single device.

## Smoke tests

```bash
python -m compileall src train.py evaluate.py predict.py
pytest

# 1-epoch end-to-end, small and offline
python train.py model=smp/unet train.epochs=1 train.batch_size=2 train.num_workers=0 \
    train.img_size=128 model.encoder_weights=null hardware.accelerator=cpu
```

> Override the dataset location with `dataset.root=<path>` if needed.
> `model.encoder_weights=null` avoids downloading ImageNet weights.

## Reference

**Optimizer / scheduler** (`configs/optimizer/*.yaml`): `optimizer.lr`,
`optimizer.weight_decay`, `scheduler.name` (`cosine`, `step`, `plateau`, `none`).

**Logs**: `tensorboard --logdir outputs/`

**SMP architectures**: `Unet` · `UnetPlusPlus` · `MAnet` · `Linknet` · `FPN` ·
`PSPNet` · `DeepLabV3` · `DeepLabV3Plus` · `PAN`.
Encoder list: https://smp.readthedocs.io/en/latest/encoders.html
