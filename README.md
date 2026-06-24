# seg-baseline

A lightweight, config-driven research baseline for 2D medical image segmentation.
Built on [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) (SMP), [PyTorch Lightning](https://lightning.ai/), and [Hydra](https://hydra.cc/).

**Key features**

- **Config-driven**: swap model / backbone / loss / optimizer / augmentation from the command line — no code changes.
- **Pluggable model providers**: SMP baselines and a custom `ModularUNet` through a single `build_model` factory. (TransUNet is *planned* — see the status table below.)
- **Multi-class & binary segmentation** through one code path.
- **Explicit evaluation CSV**: `summary.csv` / `per_class.csv` / `per_case.csv` written on every `evaluate.py` run (not just TensorBoard).
- **Unified model output contract**: models may return a logits tensor or a dict; `normalize_model_output` handles both.
- **Multi-task scaffold (Phase 9b)**: optional, eval-only boundary and clinical-morphology metrics — all OFF by default.

---

## Status: implemented vs planned

| Capability | Status |
|---|---|
| SMP baselines (`model=smp/*`, legacy `model=unet`) | ✅ implemented |
| `build_model` factory + `provider` field | ✅ implemented |
| Loss factory (`dice_ce`, `dice_bce`) | ✅ implemented |
| Output contract `normalize_model_output` (Phase 9a) | ✅ implemented |
| Explicit eval CSV: summary / per_class / per_case (Phase 4) | ✅ implemented |
| `ModularUNet` + identity skip (Phase 6) | ✅ implemented |
| Clinical morphology eval metrics (`clinical_metrics.csv`) | ✅ implemented (eval-only, **pixel units only**) |
| Boundary eval metrics (`boundary_metrics.csv`) | ✅ implemented (derived from the seg prediction) |
| `BoundaryHead` / `ClinicalHead`, `MultiTaskLoss` | 🟡 skeleton — present but **not wired** into training |
| Extra skip modules (scse, cbam, …) | 🟡 planned — registry ready, only `identity` shipped |
| **TransUNet** (`model=custom/transunet`) | ⛔ **planned / not implemented** — the command will error today |
| Physical-unit (mm/mm²) clinical metrics | ⛔ planned — only pixel units are computed |
| DDP-safe per-case CSV merge | ⛔ planned — CSV export is single-process |

---

## Project structure

```
Medical-Image-Segmentation-Baseline/
├── configs/
│   ├── train.yaml                 ← main config (defaults: model/dataset/aug/optimizer/loss/task)
│   ├── model/
│   │   ├── unet.yaml …            ← legacy configs (no `provider` → treated as smp)
│   │   ├── smp/                   ← smp/unet, smp/unetplusplus, smp/deeplabv3plus, smp/fpn, smp/manet
│   │   └── custom/
│   │       └── modular_unet.yaml  ← ModularUNet (custom provider)
│   ├── dataset/  augmentation/  optimizer/
│   ├── loss/                      ← dice_ce, dice_bce
│   └── task/default.yaml          ← multi-task output switches (all OFF)
├── src/
│   ├── datasets/   seg_dataset.py (dict batch + metadata), batch.py (unpack_batch)
│   ├── models/
│   │   ├── factory.py             ← build_model(cfg.model, cfg.dataset)
│   │   ├── smp_models.py          ← build_smp_model (only place smp models are built)
│   │   ├── outputs.py             ← normalize_model_output (output contract)
│   │   ├── seg_module.py          ← LightningModule (provider-agnostic)
│   │   ├── heads/                 ← BoundaryHead / ClinicalHead (skeleton)
│   │   └── modular_unet/          ← encoder/decoder/blocks/fusion/skip_modules
│   ├── losses/                    ← factory.py, segmentation_losses.py, multitask_loss.py (skeleton)
│   ├── metrics/                   ← functional.py, evaluator.py, morphology.py, boundary_metrics.py, clinical_metrics.py
│   ├── tasks/                     ← segmentation / boundary / clinical_morphology helpers
│   ├── transforms/   utils/ (io.py, geometry.py)
├── train.py   evaluate.py   predict.py
├── tests/test_smoke.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `segmentation-models-pytorch`, `albumentations`, `hydra-core`, `torchmetrics`. For tests: `pytest`.

---

## Models — how to run

### SMP baselines

```bash
# new namespace
python train.py model=smp/unet
python train.py model=smp/unetplusplus model.encoder=resnet50
python train.py model=smp/deeplabv3plus model.encoder=resnet50

# legacy configs still work (no `provider` field → defaults to smp)
python train.py model=unet
```

### ModularUNet (custom, identity skip)

A controllable UNet (SMP encoder backbone + our own decoder) for skip-connection
research. v1 ships only the `identity` skip module as an ablation control.

```bash
python train.py model=custom/modular_unet model.skip.name=identity
```

> `model.img_size` must be divisible by 32 (resnet-style 5-stage encoder).

### TransUNet — ⛔ planned (not implemented)

TransUNet is **not** implemented yet: there is no `configs/model/custom/transunet.yaml`
and the factory has no `transunet` branch, so the command below will currently
**fail** (Hydra cannot find the config group). It is documented here only as the
target API for Phase 5:

```bash
# PLANNED — not functional yet
python train.py model=custom/transunet train.img_size=224
```

---

## Model config namespaces & migration

Each model config carries a `provider` field consumed by `build_model`:

```yaml
# configs/model/smp/unet.yaml
provider: smp          # "smp" or "custom"
name: unet
arch: Unet             # SMP architecture name (also used as the run-name label)
encoder: resnet34
encoder_weights: imagenet   # null for random init
in_channels: 3
params: {}             # extra kwargs forwarded to smp.create_model
```

**Migration from the old layout** (no breaking change required):

- Old configs such as `configs/model/unet.yaml` have **no** `provider` field. They
  still work — `build_model` treats a missing `provider` as `smp`. So
  `python train.py model=unet` is unchanged.
- The new `configs/model/smp/*.yaml` are the preferred namespace going forward
  (`model=smp/unet`). Custom models live under `configs/model/custom/`.
- Any custom model config must include an `arch` label field because the run name
  is `logging.name = ${model.arch}_${model.encoder}_${dataset.name}`.

---

## Loss configuration

Loss is built by `build_loss(cfg.loss, cfg.dataset)` (`src/losses/factory.py`).
The task type is derived from `dataset.num_classes`:

| Loss | Task | Composition |
|---|---|---|
| `dice_ce` (default) | multiclass (`num_classes > 1`) | Dice + CrossEntropy |
| `dice_bce` | binary (`num_classes == 1`) | Dice + BCEWithLogits |

```bash
python train.py loss=dice_ce        # default
python train.py dataset=binaryclass loss=dice_bce
```

> The default `loss=dice_ce` matches the default multiclass dataset. For a binary
> dataset you must pass `loss=dice_bce`; using `dice_ce` on a binary task raises a
> clear error. All losses are computed from logits (models never apply
> sigmoid/softmax). If `cfg.loss` is absent (e.g. an old checkpoint), the loss
> falls back to the previous default (Dice + CE/BCE by task).

---

## Dataset

```
your-dataset/
├── train/
│   ├── images/          ← RGB images (.png / .jpg)
│   └── <mask_dir>/      ← segmentation masks (.png, same stem as the image)
├── val/   …
└── test/  …
```

| `mask_mode` | pixel values | use case |
|---|---|---|
| `index`  | 0, 1, 2, … (class indices) | multi-class |
| `binary` | 0 / 255 → auto-converted to 0 / 1 | binary |

### Binary vs multiclass — what differs

| | binary | multiclass |
|---|---|---|
| `num_classes` | 1 | > 1 |
| model output | `B×1×H×W` | `B×num_classes×H×W` |
| prediction | `sigmoid(logits) > eval.threshold` | `argmax(logits, dim=1)` |
| loss | `dice_bce` | `dice_ce` |
| mean metrics | the single foreground (mask value 1) | over `dataset.foreground_classes`, background excluded by default |

### Add a new dataset

Copy `configs/dataset/multiclass.yaml` (or `binaryclass.yaml`), edit, then:

```bash
python train.py dataset=my_dataset
```

```yaml
name: my_dataset
root: /path/to/dataset
mask_dir: masks
mask_mode: index
num_classes: 3
class_names: [background, class_a, class_b]
foreground_classes: [1, 2]   # indices used for mean Dice/IoU (excludes background)
```

---

## Training

```bash
python train.py                       # defaults: UNet + ResNet34, multiclass, dice_ce
python train.py train.batch_size=4 train.epochs=200 train.img_size=640
python train.py optimizer=sgd optimizer.lr=1e-2 augmentation=heavy
```

Multi-GPU / DDP and Hydra multirun behave as before:

```bash
python train.py hardware.devices=4 hardware.strategy=ddp train.precision=16-mixed
python train.py --multirun model=smp/unet,smp/unetplusplus model.encoder=resnet34,resnet50
```

---

## Evaluation & CSV outputs

```bash
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=test eval.save_csv=true
python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt split=val
```

Besides the printed metric table, explicit research CSVs are written to
`outputs/<logging.name>/eval/<split>/`:

| file | one row per | key columns |
|---|---|---|
| `summary.csv` | evaluation run | `num_cases, mean_dice, mean_iou, mean_precision, mean_recall` (+ run/model/encoder) |
| `per_class.csv` | class | `class_id, class_name, dice, iou, precision, recall, support_pixels` |
| `per_case.csv` | case | `case_id, image_path, mask_path, dice_mean, iou_mean, precision_mean, recall_mean, pred_path` (+ per-class columns) |
| `config.yaml` | — | the fully-resolved config |

`eval.*` switches (in `configs/train.yaml`):

```yaml
eval:
  save_csv: true
  threshold: 0.5            # binary prediction threshold (per-case CSV)
  include_background: false # include background in mean_* if true
  output_dir: null          # default: <save_dir>/<logging.name>/eval/<split>
  per_case: true
  per_class: true
  save_predictions: false   # planned — not implemented
  save_overlay: false       # planned — not implemented
```

### Metric conventions

- Aggregation is **macro**: per-case metrics are averaged across cases; `summary`
  averages over foreground classes (background excluded unless `include_background=true`).
- **Empty-mask convention** (avoids NaN; medical scans often miss a class):
  - pred empty **and** target empty → `dice = 1, iou = 1`
  - pred non-empty **and** target empty → `dice = 0, iou = 0`
  - pred empty **and** target non-empty → `dice = 0, iou = 0`
  - precision / recall follow the same rule (a fully-empty pred/target pair scores 1).
- The CSV `mean_dice` may differ slightly from the logged `test/dice_mean`: the CSV
  uses per-case macro averaging with the empty-mask convention, while the logged
  value uses torchmetrics' global aggregation. The CSV is the research artifact.
- Lightning's `CSVLogger` `metrics.csv` is still produced (in its own
  `outputs/<name>_eval_<split>/` directory) and is kept for convenience.

---

## Multi-task outputs (Phase 9b) — optional, OFF by default

The segmentation baseline is unaffected unless you explicitly enable a task.

### Output contract (Phase 9a)

Every model's forward output passes through `normalize_model_output` (`src/models/outputs.py`):

- a bare `Tensor` → `{"seg_logits": tensor}` (no dtype/device/shape change, no activation);
- a `dict` must contain `"seg_logits"` (else a clear `ValueError`);
- anything else → `TypeError`.

This lets future models emit `{"seg_logits": ..., "boundary_logits": ..., "clinical": ..., "features": ...}` without changing `SegModule`.

### Clinical morphology metrics (eval-only, deterministic)

```bash
python evaluate.py checkpoint=... split=test \
    task.outputs.clinical.enabled=true
# optional: task.outputs.clinical.metrics=[area,perimeter,area_ratio]
```

Writes `clinical_metrics.csv` (one row per case) computed **from the predicted mask**.

- **Units are pixels only** — every row has a `unit=pixel` column. No physical
  (mm / mm²) conversion is performed; that is *planned*, not implemented.
- Non-differentiable; used for evaluation only — never inside a training loss.

### Boundary metrics (eval-only)

```bash
python evaluate.py checkpoint=... split=test \
    task.outputs.boundary.enabled=true \
    task.outputs.boundary.boundary_width_px=3
```

Writes `boundary_metrics.csv` (boundary dice/iou per foreground class + means),
computed by comparing the boundary band of the **segmentation prediction** vs the
target (via `mask_to_boundary`). It does **not** use a learned boundary output —
`BoundaryHead` and boundary-loss training are skeletons only.

> Skeletons present but **not wired** into training: `src/models/heads/BoundaryHead`,
> `ClinicalHead`, and `src/losses/multitask_loss.MultiTaskLoss` (defaults to
> segmentation-only). They exist so a future boundary/clinical *training* pipeline
> can adopt them without touching `SegModule`.

---

## Single-image inference

```bash
python predict.py --img path/to/image.png \
    --checkpoint outputs/<name>/checkpoints/best.ckpt --out result.png
```

Produces a side-by-side input / colour-coded prediction visualisation.

---

## Developer guide

### Add a new baseline model

1. Implement the model under `src/models/<new_model>/` (a `torch.nn.Module`).
2. Register it in `src/models/factory.py` under `provider == "custom"` (dispatch on `name`).
3. Add `configs/model/custom/<new_model>.yaml` with `provider: custom`, `name: <new_model>`,
   and an `arch:` label (used by `logging.name`).
4. Ensure `forward` returns logits `B×C×H×W` **or** a dict containing `"seg_logits"`
   (both are handled by `normalize_model_output`).
5. Add a forward-shape test (see `tests/test_smoke.py::test_modular_unet_identity_forward_shape`).

### Add a new skip module (ModularUNet)

1. Implement an `nn.Module` in `src/models/modular_unet/skip_modules.py`:

   ```python
   class MySkip(SkipModule):
       def __init__(self, channels_by_level=None, **params): ...
       def forward(self, skip, decoder_feature=None, level: int = 0):
           return skip  # must preserve spatial size; v1 also preserves channels
   ```

2. Register it: `SKIP_MODULES["my_skip"] = MySkip`.
3. Select it via `model.skip.name=my_skip` (no config file needed).
4. Run the identity-vs-new-module ablation:

   ```bash
   python train.py --multirun model=custom/modular_unet \
       model.skip.name=identity,my_skip train.seed=42,43,44
   ```

> v1 fusion supports only `placement=before_concat` + `mode=concat`; other modes
> raise `NotImplementedError` (TODO).

---

## Smoke tests

```bash
python -m compileall src train.py evaluate.py predict.py
pytest                                   # unit + smoke tests

# 1-epoch end-to-end (small/offline)
python train.py model=smp/unet train.epochs=1 train.batch_size=2 train.num_workers=0 \
    train.img_size=128 model.encoder_weights=null hardware.accelerator=cpu

python train.py model=custom/modular_unet model.skip.name=identity \
    train.epochs=1 train.batch_size=2 train.num_workers=0 train.img_size=128 \
    model.encoder_weights=null hardware.accelerator=cpu

python evaluate.py checkpoint=outputs/<name>/checkpoints/best.ckpt \
    split=test eval.save_csv=true
```

> The smoke commands assume the dataset configured in `configs/dataset/multiclass.yaml`
> is reachable; override its location with `dataset.root=<path-to-dataset>` if needed.
> `model.encoder_weights=null` avoids downloading ImageNet weights.

---

## Configuration reference

**`configs/optimizer/*.yaml`**

| Parameter | Example | Description |
|---|---|---|
| `optimizer.lr` | `1e-4` | Learning rate |
| `optimizer.weight_decay` | `1e-4` | Weight decay |
| `scheduler.name` | `cosine` | `cosine`, `step`, `plateau`, `none` |

**Viewing logs**

```bash
tensorboard --logdir outputs/
```

## Supported SMP architectures

`Unet` · `UnetPlusPlus` · `MAnet` · `Linknet` · `FPN` · `PSPNet` · `DeepLabV3` · `DeepLabV3Plus` · `PAN`
Full encoder list: https://smp.readthedocs.io/en/latest/encoders.html
