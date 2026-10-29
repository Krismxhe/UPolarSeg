# UPolarSeg

Research code for **2D medical image segmentation and polar quantification of the optic disc (OD) and peripapillary atrophy (PPA) in fundus images**. The project provides a shared data-loading, training, and per-case evaluation workflow for Cartesian and polar inputs, SMP and TransUNet baselines, and ModularUNet research configurations.

Main features:

- Multiclass semantic segmentation, with default labels `background=0`, `OD=1`, and `PPA=2`.
- Independent hard and soft readouts for PMW, Dv, and PVDR, with fixed-axis and maximum-center-chord definitions of Dv.
- Optional polar and Cartesian prediction masks, hard and soft measurement overlays, and hard GT overlays.
- Optional OD/PPA contours and hard PMW, Dv, and Dh measurements over fundus images.
- Separate OD and PPA FP/FN error maps and corresponding fundus error overlays.
- Per-case OD/PPA metrics, patient-clustered statistics, confidence intervals, and paired comparisons.

This README describes the current code and configurations. Replace example paths, checkpoints, and GPU indices with values for your environment. Sections 3.3 and 10 document a current ModularUNet constructor mismatch; successful configuration parsing does not establish that those examples can run.

Quick links: [Training](#3-training) · [Evaluation](#4-checkpoint-evaluation) · [Export options](#6-masks-measurement-overlays-and-error-maps) · [Checks](#9-reproducibility-and-checks).

## 1. Environment and Installation

The pinned dependencies require Python ≥3.11. Run from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](requirements.txt) pins versions from the project environment and includes `rich`, `monai`, `pytest`, `ml-collections`, and `scipy`. The separate installation steps for these packages in the previous README are no longer needed.

| Component | Version in requirements |
| --- | --- |
| PyTorch / TorchVision | `2.12.1 / 0.27.1` |
| PyTorch Lightning / TorchMetrics | `2.6.5 / 1.9.0` |
| segmentation-models-pytorch / timm | `0.5.0 / 1.0.27` |
| Hydra / OmegaConf | `1.3.3 / 2.3.1` |
| Albumentations / albucore | `1.4.24 / 0.0.23` |
| NumPy | `2.4.6` |
| OpenCV | `opencv-python-headless==4.11.0.86` |

Match the Python minor version and PyTorch build used for training. Recent functional checks used Python 3.12 and CPU PyTorch. The version numbers above do not fully specify the CUDA wheel source, driver, or GPU environment; requirements is a project dependency list, not a complete machine snapshot.

Keep only one OpenCV distribution that provides `cv2`. The current requirements use the headless package; do not install both `opencv-python` and `opencv-python-headless`.

For SMP and ModularUNet, `model.encoder_weights=imagenet` may download pretrained weights. Set it to `null` for training from scratch or offline initialization. See Section 10 for the separate TransUNet pretrained-weight entry point and configuration limitations.

## 2. Data, Labels, and Coordinates

### 2.1 Dataset Layout

Each split requires images and PNG masks with matching filename stems:

| Content | Example paths |
| --- | --- |
| Training image / mask | `<dataset_root>/train/images/case001.jpg`, `<dataset_root>/train/masks_semantic/case001.png` |
| Validation image / mask | `<dataset_root>/val/images/case101.jpg`, `<dataset_root>/val/masks_semantic/case101.png` |
| Test image / mask | `<dataset_root>/test/images/case201.jpg`, `<dataset_root>/test/masks_semantic/case201.png` |

`dataset.mask_dir` selects the mask subdirectory. Supported image extensions are PNG, JPG/JPEG, TIF/TIFF, and BMP; masks must be PNG files. The filename stem becomes `case_id`, so a split cannot contain both `case001.jpg` and `case001.png` as separate input images.

| Task | Configuration | Mask pixel values |
| --- | --- | --- |
| Three-class OD/PPA segmentation | `mask_mode=index`, `num_classes=3` | `0=background, 1=OD, 2=PPA` |
| Binary segmentation | `dataset=binaryclass loss=dice_bce` | Input GT uses `0/255`; the loader applies `>127` to produce `0/1` |

Index masks are not automatically remapped from `0/127/255` to `0/1/2`. Class-index PNG files and RGB visualization images are not interchangeable.

`train.py` fits on `train` and `val`, then automatically evaluates the best checkpoint on `test`, so the complete training workflow requires all three splits. `evaluate.py` loads only the requested split and uses evaluation transforms without random crops or flips, including for `split=train`.

**`predict_masks.py` still uses the same dataloader and requires paired GT. It is not an inference entry point for an unlabeled image folder.**

### 2.2 Dataset Configurations and the Polar Switch

The repository includes Beijing, Zhengzhou, and Tibet dataset configurations, for example:

- `dataset=beijing_1024_cart`, `dataset=beijing_1024_polar`
- `dataset=zhengzhou_1024_cart`, `dataset=zhengzhou_1024_polar`
- `dataset=tibet_1024_cart`, `dataset=tibet_1024_polar`

These configurations contain machine-specific default `root` paths. Set `dataset.root=/path/to/dataset` explicitly for each run.

For **`train.py` and `evaluate.py`**, a dataset name ending in `_polar` **does not enable polar transforms automatically**. Add `polar=enabled`. Input files should still be original Cartesian images or ROIs and their masks; the online transform creates the polar grid after resizing. Do not pass already-unwrapped polar images through the same online polar transform again.

The default `--input_space auto` in `predict_masks.py` infers the input space from the dataset name. The examples here explicitly use `--input_space polar` or `cartesian` to make the setting clear.

### 2.3 Shared Geometry and Image Size

Geometry is defined by [configs/polar/enabled.yaml](configs/polar/enabled.yaml) and [polar_transforms.py](src/transforms/polar_transforms.py):

| Parameter | Default behavior |
| --- | --- |
| `train.img_size` | Resize the Cartesian image to a square with this side length |
| `polar.geometry.center=null` | Use the resized-image center `((W-1)/2, (H-1)/2)` in `(x,y)` order |
| `polar.geometry.pt_type=corner`, `max_radius=null` | Use the distance from the selected center to the farthest image corner |
| `radius_bins`, `angle_bins` | Both follow `train.img_size` by default |
| `polar.geometry.mode` | Default: `linear` |
| Polar image layout | Height represents angle bins; width represents radius bins. The left edge at `r=0` maps to the same center |

For a 512-pixel input, the default center is `(255.5,255.5)`, not `(256,256)`. Resolve geometry from the configuration after changing input size instead of retaining a hardcoded radius of 362. The center is not automatically replaced by the OD centroid.

Class-mask geometry transforms use nearest-neighbor sampling. Quantification readouts currently require **three-class index masks and linear polar input**. Segmentation, mask export, and error maps support a broader range of inputs; these quantification restrictions do not apply to every feature.

## 3. Training

### 3.1 Cartesian TransUNet Baseline

```bash
python train.py \
  model=baseline/transunet \
  dataset=beijing_1024_cart \
  dataset.root=/path/to/beijing \
  polar=disabled \
  augmentation=medium \
  train.img_size=512 \
  train.batch_size=4 \
  'hardware.devices=[0]' \
  logging.name=transunet_beijing_cart
```

TransUNet's `model.img_size` follows `train.img_size`. The current preset uses patch size 16, and the input dimensions must satisfy the model's grid requirements. Adjust batch size to available GPU memory. This command does not enable an auxiliary quantification loss.

### 3.2 Polar DeepLabV3+ Baseline

```bash
python train.py \
  model=deeplabv3plus \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  train.batch_size=4 \
  'hardware.devices=[4,5]' \
  logging.name=deeplabv3plus_beijing_polar
```

`model=deeplabv3plus` uses ResNet34 in the current configuration. Select `model=deeplabv3plus_resnet50` for ResNet50, or use another existing configuration such as `smp/unet` or `smp/unetplusplus`.

Polar input and auxiliary quantification training are independent choices. The command above trains segmentation only. To apply an auxiliary quantification loss directly to `seg_logits`, add `+quantification=pmw_pvdr`; its defaults are `adapter_enabled=false` and `weight=0.05`.

### 3.3 ModularUNet PMSR-lite and the Quantification Adapter

**Current prerequisite:** the [ModularUNet builder](src/models/research/modular_unet/model.py) passes `boundary_enabled` to a constructor that does not accept it, causing `unexpected keyword argument 'boundary_enabled'`. The example below shows the intended configuration and requires that interface mismatch to be fixed first. If your local version already includes the fix, check it against your implementation.

```bash
python train.py \
  model=research/modular_unet_pmsr_lite \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  train.batch_size=8 \
  'hardware.devices=[4,5]' \
  +quantification=pmw_pvdr \
  quantification.adapter_enabled=true \
  quantification.dv_mode=fixed_vertical \
  logging.name=pmsr_lite_adapter_beijing_polar
```

The quantification group is defined in [configs/quantification/pmw_pvdr.yaml](configs/quantification/pmw_pvdr.yaml) and is absent from the main configuration's defaults. Use `+quantification=pmw_pvdr` to add it. Do not add it again if an experiment configuration already loads it.

| Training configuration | Behavior |
| --- | --- |
| No quantification group, or `quantification.enabled=false` | Segmentation loss only |
| `enabled=true, adapter_enabled=false` | Segmentation loss plus an auxiliary quantification loss on `seg_logits`; quantification gradients can update the segmentation model |
| `enabled=true, adapter_enabled=true` | Segmentation loss updates the segmentation model; the adapter receives detached features and logits and is updated by quantification loss |

The adapter requires a model that accepts `return_features=True`, returns `features.decoder_final`, and has a matching decoder-channel configuration. Setting `adapter_enabled=true` alone does not make arbitrary SMP or TransUNet baselines compatible. ModularUNet currently provides this feature interface.

### 3.4 Common Training Settings and Outputs

| Setting | Example / behavior |
| --- | --- |
| Learning rate | `optimizer.lr=1e-4`, not `train.lr` |
| Epochs / seed | `train.epochs=150 train.seed=42` |
| Precision | `train.precision=32-true` or `16-mixed` |
| Batch size per process | `train.batch_size=8` means 8 per process, not 8 across all GPUs |
| GPU selection | `'hardware.devices=[4,5]'`; indices refer to the currently visible devices |
| Device count | `hardware.devices=1` requests one device; it does not select GPU 1 |
| Checkpoint monitor | Defaults: `checkpoint.monitor=val/dice_mean`, `checkpoint.mode=max` |
| Run name | Set `logging.name=<experiment_name>` to distinguish experiments |

Default TensorBoard logs and checkpoints are stored under:

```text
outputs/<logging.name>/version_<n>/
outputs/<logging.name>/version_<n>/checkpoints/best.ckpt
outputs/<logging.name>/version_<n>/checkpoints/last.ckpt
```

At the end of training, `train.py` calls `trainer.test(..., ckpt_path='best')`. Enabled image exports run during this stage. Use `evaluate.py` for explicit per-case research CSV exports; the final Lightning test log is not a complete research report.

## 4. Checkpoint Evaluation

### 4.1 Match the Training Configuration

**`evaluate.py` rebuilds the model from the supplied configuration; it does not automatically restore the checkpoint's complete training configuration.** Loading uses `strict=False`, so a successful load alone does not establish that the architecture, adapter, and input geometry match.

Check `model` and its structural parameters, class definitions, `train.img_size`, normalization, `polar` geometry, and adapter settings. Record the Dv definition used for auxiliary quantification. For evaluation on another dataset, change `dataset` and `dataset.root` while preserving the model's input space and label meanings.

Training and evaluation share `configs/train.yaml`. During training, `checkpoint` is a callback configuration node; during evaluation, `checkpoint=/path/to/best.ckpt` replaces it with the checkpoint path.

### 4.2 Cartesian TransUNet: External Evaluation on Zhengzhou

```bash
python evaluate.py \
  checkpoint=/path/to/transunet_beijing_best.ckpt \
  model=baseline/transunet \
  dataset=zhengzhou_1024_cart \
  dataset.root=/path/to/zhengzhou \
  polar=disabled \
  augmentation=medium \
  train.img_size=512 \
  'hardware.devices=[0]' \
  split=test \
  eval.odppa.enabled=true \
  eval.output_dir=outputs/eval/transunet_zhengzhou_cart
```

`eval.odppa.enabled=true` enables OD/PPA research evaluation on the native Cartesian GT grid, including DSC, NSD, and HD95. Cartesian models support this segmentation evaluation. Polar quantification readouts are unavailable for Cartesian input, and the reason is recorded.

### 4.3 Polar Segmentation Baseline: Measurement Without Auxiliary Quantification Loss

For the DeepLabV3+ checkpoint trained without an adapter in Section 3.2:

```bash
python evaluate.py \
  checkpoint=/path/to/deeplabv3plus_polar_best.ckpt \
  model=deeplabv3plus \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  'hardware.devices=[4]' \
  split=test \
  +quantification=pmw_pvdr \
  quantification.enabled=false \
  quantification.dv_mode=fixed_vertical \
  eval.odppa.enabled=true \
  eval.output_dir=outputs/eval/deeplabv3plus_beijing_polar
```

Here, `quantification.enabled=false` disables the auxiliary loss, while `eval.odppa.enabled=true` still builds an independent readout for eligible polar data and exports hard and soft measurements. Retraining is not required to obtain these measurements.

### 4.4 Evaluate a PMSR Checkpoint Trained with an Adapter

This configuration matches Section 3.3 and also requires the ModularUNet constructor mismatch to be fixed:

```bash
python evaluate.py \
  checkpoint=/path/to/pmsr_lite_adapter_best.ckpt \
  model=research/modular_unet_pmsr_lite \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  'hardware.devices=[4]' \
  split=test \
  +quantification=pmw_pvdr \
  quantification.adapter_enabled=true \
  quantification.dv_mode=fixed_vertical \
  eval.odppa.enabled=true \
  eval.output_dir=outputs/eval/pmsr_lite_adapter_beijing_polar
```

Do not use the previous example's `quantification.enabled=false` for a checkpoint trained with an adapter: the current entry point would omit the adapter. For a checkpoint without an adapter, keep `adapter_enabled=false`.

Each command needs `polar=enabled` only once. In Bash, a continuation backslash `\` must be the final character on its line. Omit it from the last line, and do not place additional arguments on a new line after the command has ended.

### 4.5 Inspect Configuration Without Running Training or Evaluation

Hydra can print the composed configuration:

```bash
python train.py --cfg job --resolve \
  model=deeplabv3plus \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  'hardware.devices=[4,5]'
```

You can also add `--cfg job --resolve` to an evaluation command. This checks configuration composition and interpolation only; it does not verify model construction, checkpoint compatibility, GPU memory, or the presence of required data files.

## 5. PMW, Dv, Dh, and PVDR

### 5.1 Hard and soft

Both readouts use the same polar geometry. They differ in how they obtain the OD radius and PPA width at each angle:

- **Hard**: reads radial boundaries from the class-index mask and checks geometric validity, including the origin, radial continuity, and clipping.
- **Soft**: integrates OD/PPA probabilities along each radius to obtain equivalent occupancy lengths. This is a differentiable surrogate readout.

Hard and soft may select different angles for maximum PMW. In maximum center chord mode, they may also select different Dv directions. Soft line segments do not necessarily lie on the argmax mask boundary. An argmax PNG alone cannot recover the original soft readout; the inference logits or probabilities are required.

### 5.2 Two Dv modes and perpendicular Dh

Let `r_OD(theta)` be the OD radius from the selected readout and `w_PPA(theta)` its radial PPA width:

| Measurement | Definition |
| --- | --- |
| PMW | `max_theta w_PPA(theta)` |
| `dv_mode=fixed_vertical` | `r_OD(theta0) + r_OD(theta0 + 180°)` |
| `dv_mode=max_center_chord` | `max_theta [r_OD(theta) + r_OD(theta + 180°)]` |
| Dh | Rotate the selected Dv direction by 90° and sum the two opposite OD radii on that axis |
| PVDR | `PMW / max(Dv, denominator_floor_px)` |

The defaults are `dv_mode=fixed_vertical` and `vertical_angle_deg=90.0`: Dv uses 90°/270°, and Dh uses 0°/180°. To select the maximum center chord:

```yaml
quantification:
  dv_mode: max_center_chord
```

Merge this field into the loaded quantification configuration. On the CLI, first load `+quantification=pmw_pvdr`, then override `quantification.dv_mode=max_center_chord`.

The maximum center chord compares the sums of opposite radii across all directions. It is neither `2 * max(radius)` nor the maximum distance between arbitrary boundary points. Dh rotates with the selected Dv; it does not independently search for a second longest chord. `vertical_angle_deg` affects only the fixed-axis mode.

In a polar image, a center chord appears as two horizontal radius segments separated by 180°. In a Cartesian image, it is a single line through the center. Directions between sampled angle rows use periodic linear interpolation.

**Lengths always use resized Cartesian ROI pixels, `resized_roi_px`.** Restoring the original image size changes only the display coordinates; it does not convert measurements to original-image pixels or millimeters. The default denominator floor is `denominator_floor_px=1.0`. Dh is currently derived for visualization and stored in overlay JSON; it has not been added to the original PMW/Dv/PVDR loss or CSV metric schema.

The fixed-axis and maximum center chord modes use the `polar-radial-v1` and `polar-radial-v2` protocols, respectively. Model comparisons must use consistent Dv definitions, geometry, and units; results from the two protocols must not be pooled directly. See [DV_MODES.md](docs/DV_MODES.md).

## 6. Masks, Measurement Overlays, and Error Maps

All image export switches in the main evaluation configuration are disabled by default, and their controls are independent. They are available in `evaluate.py` and the final test stage of `train.py`. Saving images does not require `eval.save_csv=true`.

### 6.1 Export controls

| Configuration | Default | Options / output |
| --- | --- | --- |
| `eval.save_predictions` | `none` | `none / polar / cartesian / both`; class-index masks |
| `eval.save_quantification_overlays` | `none` | `none / polar / cartesian / both`; prediction measurement overlays |
| `eval.save_quantification_gt_overlays` | `none` | Same options; separate ground-truth (GT) hard measurement overlays |
| `eval.save_fundus_overlays` | `none` | `none / prediction / groundtruth / both`; hard measurements on fundus images |
| `eval.save_error_maps` | `none` | `none / polar / cartesian / both`; categorical FP/FN maps |
| `eval.save_error_fundus` | `false` | `true / false`; independent fundus FP/FN overlays |
| `eval.prediction_cartesian_size` | `original` | `original / input`; shared size control for Cartesian and fundus exports |

- `original`: restore the original dimensions of the current dataset image/GT.
- `input`: retain the resized Cartesian input dimensions, such as 512×512.
- Polar exports retain the selected `angle_bins × radius_bins` grid.
- If the dataset image is already a cropped ROI, `original` refers to that ROI. It does not place the result back into a separate full-frame fundus image.
- Class-index PNGs preserve label values such as `0/1/2`, so they may look nearly black. RGB overlays are saved separately.
- The legacy `eval.save_overlay` field is not implemented. Use the controls above.

### 6.2 Separate or combined hard/soft images

`eval.quantification_overlay_readouts` selects the readouts to draw, and `eval.quantification_overlay_layout` selects how to save them:

| readouts | layout | Output in each coordinate space |
| --- | --- | --- |
| `hard` | `combined` | One image containing only hard measurements |
| `soft` | `combined` | One image containing only soft measurements |
| `both` | `separate` | One hard image and one soft image |
| `both` | `combined` | One image containing both hard and soft measurements |

The defaults are `readouts=both` and `layout=combined`. A single readout can also use `separate`, which creates only its corresponding subdirectory. Hard measurements use solid lines; soft measurements use dashed lines. Each readout includes PMW, Dv, and perpendicular Dh. Set `eval.quantification_overlay_legend=false` to disable the legend.

GT uses its own hard readout and is saved independently through `save_quantification_gt_overlays`, without being mixed into prediction hard/soft images. Invalid hard measurements are not drawn as valid measurement lines; JSON retains their validity flags and candidate values.

### 6.3 Fundus backgrounds, contours, and hard measurements

Set `eval.save_fundus_overlays=prediction`, `groundtruth`, or `both` to save prediction images, GT images, or both.

Fundus overlays read the corresponding original Cartesian RGB image directly and add OD/PPA contours and **hard PMW, Dv, and Dh**. The default `eval.fundus_overlay_alpha=0.0` leaves the foreground unfilled and shows only contours and lines. A value such as `0.15` adds a translucent foreground fill.

Fundus measurement overlays always use hard readouts, regardless of `quantification_overlay_readouts=soft` or `layout=separate`. Measurement endpoints are transformed geometrically and the lines are redrawn; the code does not inverse-warp a polar RGB image containing text. Restoring width and height by different scale factors can change the displayed angles. Dv and Dh are defined as perpendicular on the input ROI grid.

**Measurement overlays and fundus measurement overlays currently require three-class input on a linear polar grid.** Cartesian models can export masks and error maps, but cannot directly use this polar measurement rendering pipeline.

See [QUANTIFICATION_OVERLAYS.md](docs/QUANTIFICATION_OVERLAYS.md) and [FUNDUS_OVERLAYS.md](docs/FUNDUS_OVERLAYS.md).

### 6.4 OD/PPA error maps

Each class receives its own image:

```text
FP_c = (prediction == c) & (GT != c)
FN_c = (GT == c) & (prediction != c)
```

| Pixel category | Categorical error map | Fundus error overlay |
| --- | --- | --- |
| FP, over-segmentation | Red RGB(255,0,0) | Red blended into the RGB background |
| FN, missed segmentation | Blue RGB(0,0,255) | Blue blended into the RGB background |
| Other pixels | Black | Original background, except in the legend region |

Relevant configuration:

```yaml
eval:
  save_error_maps: cartesian
  save_error_fundus: true
  error_map_classes: both       # od | ppa | both
  error_map_source: seg_logits  # seg_logits | quantification
  error_map_alpha: 0.6
  error_map_legend: true
  prediction_cartesian_size: original
```

Merge these fields into the existing `eval` node of the main configuration. Use `error_map_classes=ppa` for PPA only. For fundus error overlays only, set `save_error_maps=none` and `save_error_fundus=true`. The defaults are classes `both`, source `seg_logits`, and opacity `0.6`. Categorical error PNGs never include a legend.

Error maps do not depend on hard/soft readouts and work with Cartesian or polar models. They require valid index masks. OD/PPA class IDs come from `quantification.od_class` and `ppa_class`, defaulting to 1 and 2. Both must be distinct positive integers smaller than `num_classes`. Two-class configurations are unsupported even when exporting only OD; multiclass configurations with additional classes are supported. If OD/PPA statistics are also enabled, their class ID mapping must match.

**Comparison spaces and GT sources:**

- Cartesian/fundus errors read the original Cartesian GT from `mask_path`, align prediction and GT labels to the target dimensions, then compute FP/FN.
- Polar errors for a polar model compare the model-grid prediction with the polar GT from the batch.
- Polar errors for a Cartesian model forward-project the input-grid prediction and GT separately using the same geometry, then compare them.
- Error maps align labels before computing errors. They do not inverse-warp an already colored error image.

GT measurement overlays instead use the hard readout of the model-grid GT and project that GT into Cartesian coordinates for display. This reference pipeline differs from error maps that read the original Cartesian GT directly. Polar round-trip sampling can produce small differences between their boundaries.

`reference_complete=false` means the PPA annotation is incomplete. PPA error PNGs are skipped, and JSON marks them as not evaluable with `null` counts; OD remains evaluable. A fully annotated case with no PPA can still produce FP counts and must be distinguished from an unannotated case. See Section 8 for metadata settings.

Error JSON records the actual prediction source, comparison grid, geometry, TP/FP/FN/TN, and validity. Counts are pixels on the comparison grid and should not be compared directly across resolutions or polar/Cartesian grids. See [ERROR_MAPS.md](docs/ERROR_MAPS.md).

### 6.5 Keep prediction sources aligned

| Output | Prediction source |
| --- | --- |
| Standard prediction masks and segmentation metrics | `seg_logits` |
| Prediction measurement overlays and prediction fundus measurement overlays | `eval.quantification_overlay_source`, default `seg_logits` |
| Prediction error maps | `eval.error_map_source`, default `seg_logits` |
| Quantification evaluation | `quant_logits` when the adapter is enabled; otherwise `seg_logits` |

Both source selectors accept `seg_logits / quantification`. `quantification` selects the quantification branch actually used during evaluation: `quant_logits` when an adapter is enabled, otherwise `seg_logits`. JSON records the actual source.

Use `seg_logits` when comparing a segmentation mask with its errors. To inspect the adapter branch's boundaries and measurements, set the corresponding source to `quantification`. These selectors do not change the source of standard prediction masks.

### 6.6 Full export example: polar baseline without an adapter

The following extends the evaluation in Section 4.3. It selects maximum center chord Dv and saves masks in both spaces, separate hard/soft images, Cartesian GT measurement overlays, fundus measurement overlays, and Cartesian error maps:

```bash
python evaluate.py \
  checkpoint=/path/to/deeplabv3plus_polar_best.ckpt \
  model=deeplabv3plus \
  dataset=beijing_1024_polar \
  dataset.root=/path/to/beijing \
  polar=enabled \
  augmentation=medium_polar \
  train.img_size=512 \
  'hardware.devices=[4]' \
  split=test \
  +quantification=pmw_pvdr \
  quantification.enabled=false \
  quantification.dv_mode=max_center_chord \
  eval.odppa.enabled=true \
  eval.save_predictions=both \
  eval.prediction_cartesian_size=original \
  eval.save_quantification_overlays=both \
  eval.quantification_overlay_readouts=both \
  eval.quantification_overlay_layout=separate \
  eval.save_quantification_gt_overlays=cartesian \
  eval.save_fundus_overlays=both \
  eval.save_error_maps=cartesian \
  eval.save_error_fundus=true \
  eval.error_map_classes=both \
  eval.output_dir=outputs/eval/deeplabv3plus_polar_visuals_maxchord
```

Disable individual controls when only some outputs are needed. This example uses a different Dv protocol from the fixed-axis results in Section 4.3, so use a different output directory.

## 7. Standalone Export: predict_masks.py

`predict_masks.py` uses argparse options and accepts Hydra overrides through repeated `--override KEY=VALUE` arguments. It exports images and JSON, but does not generate the aggregate evaluation CSVs produced by `evaluate.py`.

This example saves masks and error maps for the same polar DeepLabV3+ checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 python predict_masks.py \
  --ckpt /path/to/deeplabv3plus_polar_best.ckpt \
  --model_config deeplabv3plus \
  --dataset beijing_1024_polar \
  --dataset_short beijing \
  --model_name deeplabv3plus_polar \
  --output_root outputs/predictions \
  --split test \
  --img_size 512 \
  --input_space polar \
  --save_masks both \
  --cartesian_size original \
  --save_error_maps cartesian \
  --save_error_fundus true \
  --error_map_classes both \
  --override dataset.root=/path/to/beijing \
  --override augmentation=medium_polar
```

This entry point uses one visible CUDA device, or the CPU when CUDA is unavailable. `--override hardware.devices=...` does not launch Lightning multi-GPU evaluation here. Select a GPU by setting `CUDA_VISIBLE_DEVICES` before starting the process, as in the example.

| evaluate configuration | predict_masks argument |
| --- | --- |
| `eval.save_predictions` | `--save_masks`, alias `--save_predictions` |
| `eval.prediction_cartesian_size` | `--cartesian_size` |
| `eval.save_quantification_overlays` | `--save_overlays` |
| `eval.save_quantification_gt_overlays` | `--save_gt_overlays` |
| `eval.quantification_overlay_readouts` | `--overlay_readouts` |
| `eval.quantification_overlay_layout` | `--overlay_layout` |
| `eval.quantification_overlay_source` | `--overlay_source` |
| `eval.save_fundus_overlays`, `fundus_overlay_alpha` | `--save_fundus_overlays`, `--fundus_overlay_alpha` |
| `eval.save_error_maps`, `save_error_fundus` | `--save_error_maps`, `--save_error_fundus true/false` |
| `eval.error_map_classes/source/alpha` | `--error_map_classes/source/alpha`, three separate arguments |

`--save_masks` defaults to `both`, unlike evaluate's default of `none`. `--cartesian_size` defaults to `original`, the same as evaluate. These two CLI defaults also override the corresponding `--override eval.*` settings. Other new export options retain configuration values when omitted; explicit CLI arguments take precedence. Legends can be controlled with overrides such as `--override eval.error_map_legend=false`.

Load the quantification group with `--override +quantification=pmw_pvdr`, and pass other quantification/adapter settings through separate `--override` arguments. `--max_radius` is unset by default. If supplied, it affects both the forward transform and export geometry and should match training.

## 8. Results, Metadata, and Paired Comparisons

### 8.1 Output locations

`evaluate.py` uses `eval.output_dir` as its output root. When unset, it uses:

```text
<logging.save_dir>/<logging.name>/eval/<split>/
```

This directory **does not automatically include the checkpoint's `version_<n>`**. Use separate output directories for different checkpoints, datasets, or Dv definitions. Repeated exports write the current results without clearing old images. For classes that were not evaluated, use the status and paths in the current JSON/CSV.

| Image output | Path under the evaluate output directory |
| --- | --- |
| Class-index masks | `masks_polar/<case>.png`, `masks_cartesian/<case>.png` |
| Combined prediction measurement overlays | `overlays_polar/<case>.png`, `overlays_cartesian/<case>.png` |
| Separate prediction measurement overlays | `overlays_<space>/hard/<case>.png`, `overlays_<space>/soft/<case>.png` |
| GT hard measurement overlays | `overlays_gt_polar/<case>.png`, `overlays_gt_cartesian/<case>.png` |
| Fundus hard measurement overlays | `overlays_fundus/<case>.png`, `overlays_gt_fundus/<case>.png` |
| Measurement metadata | `overlay_metadata/<case>.json` |
| Categorical error maps | `errors_<space>/od/<case>.png`, `errors_<space>/ppa/<case>.png` |
| Fundus error maps | `errors_fundus/od/<case>.png`, `errors_fundus/ppa/<case>.png` |
| Error metadata | `error_metadata/<case>.json` |

Here, `<space>` is `polar` or `cartesian`, and `<case>` is the image filename stem.

`predict_masks.py` uses `<output_root>/<model_name>` as its root and inserts `<dataset_short>` below each output directory. Standard Cartesian masks retain the legacy directory name **`masks_cart`**. For example:

```text
<output_root>/<model_name>/masks_cart/<dataset_short>/<case>.png
<output_root>/<model_name>/errors_cartesian/<dataset_short>/ppa/<case>.png
```

### 8.2 CSV files and configuration

| Condition | Output |
| --- | --- |
| Standard segmentation evaluation | `summary.csv`, plus `per_class.csv` and `per_case.csv` when enabled |
| `eval.odppa.enabled=true` | `per_case.csv`, `metrics.csv`, `summary.csv` |
| Quantification records are available | `quantification_per_case.csv`, `quantification_metrics.csv` |
| A compatible reference CSV is supplied | `comparison.csv` and/or `quantification_comparison.csv`, depending on the record type |
| Evaluation configuration | `config.yaml` |

Per-case CSVs include paths for enabled mask, overlay, and error exports. Paths are empty for images that were not selected or could not be evaluated. When exporting images with `eval.save_csv=false`, evaluate still saves the current `config.yaml`. This flag disables only the explicit research CSVs; the entry point still creates Lightning's CSVLogger.

Standard Lightning segmentation metrics and ordinary evaluation CSVs use the model input grid. OD/PPA research evaluation restores predictions to the original Cartesian GT dimensions. Polar errors, Cartesian errors, and quantification lengths use their respective grids and units; compare results on a consistent basis.

### 8.3 Patient grouping and incomplete annotations

Example metadata CSV:

```csv
case_id,patient_id,eye,reference_complete
case001,patient001,R,true
case002,patient001,L,true
case003,patient002,R,false
```

Set `eval.odppa.metadata_csv=/path/to/metadata.csv` in the evaluation command. The default cluster key is `patient_id`. Missing patient information falls back to the case identifier, so do not assume that both eyes of a patient have been grouped together.

`reference_complete=false` means **the PPA annotation is incomplete**:

- PPA research evaluation treats it as unknown/not evaluable. Unannotated regions must not be treated as negative.
- GT measurement overlays do not draw a valid PMW. OD Dv/Dh remain subject to geometric validity checks.
- PPA error images are skipped; OD error images can still be generated.

The completeness flag for a matching case in the CSV takes precedence over batch metadata. If both are missing, annotations are considered complete. Error map and GT exports can read this CSV without enabling OD/PPA statistics.

### 8.4 Paired comparisons

Add `eval.odppa.reference_csv=/path/to/baseline/per_case.csv` to the current model's evaluation command and use a separate `eval.output_dir`. Paired OD/PPA segmentation comparisons require `eval.odppa.enabled=true`; paired quantification comparisons require quantification records. Adding `reference_csv` to ordinary segmentation evaluation alone does not automatically produce a paired report. Reference results must have compatible cases, annotation statuses, and protocols. Quantification comparisons also check geometry and units.

Default statistical settings are under `eval.odppa`:

```yaml
bootstrap_samples: 10000
permutation_samples: 100000
statistics_seed: 42
cluster_key: patient_id
```

`statistics_seed` controls statistical resampling separately from `train.seed`. The current pipeline does not implement multiple-comparison correction. Interpret paired tests using the reported raw p-values and adjustment status.

The legacy `bootstrap_paired_test.py` uses a different TP/FP/FN CSV format and does not directly postprocess the current `per_case.csv`. Prefer the integrated `reference_csv` comparison workflow above for current research evaluations.

## 9. Reproducibility and Checks

### 9.1 Random seeds

The training entry point calls `pl.seed_everything(train.seed, workers=True)`, and the augmentation pipeline uses `A.Compose(seed=train.seed)`. Evaluation does not apply random training augmentations.

These settings do not guarantee bitwise reproducibility. The current Trainer does not enable a complete set of deterministic settings, and the project does not explicitly assign independent seeds to Albumentations' private random state for each worker or rank. When comparing repeated training runs, also keep software versions, GPU / CUDA, device count, batch size, worker count, precision, and data order consistent. The same seed alone does not guarantee identical weights or metrics.

### 9.2 Targeted regression checks

Quantification and Dv protocols:

```bash
python -m pytest \
  tests/test_quantification_readouts.py \
  tests/test_quantification_pipeline.py \
  tests/test_quantification_statistics.py \
  tests/test_max_center_chord.py \
  tests/test_dv_protocol.py -q
```

Masks, measurement overlays, fundus overlays, and error maps:

```bash
python -m pytest \
  tests/test_prediction_masks.py \
  tests/test_quantification_overlays.py \
  tests/test_quantification_overlay_integration.py \
  tests/test_fundus_overlays.py \
  tests/test_fundus_overlay_integration.py \
  tests/test_error_maps.py \
  tests/test_error_map_integration.py -q
```

Keep `tests/fixtures/polar_quantification_before_split.py`: it is used for numerical and gradient regression against the previous quantification implementation. These are targeted checks; they do not establish that all legacy smoke tests or end-to-end evaluations with real checkpoints have passed.

Synthetic demonstrations that require no model checkpoint or patient images:

```bash
python quick_check_quantification.py
python quick_check_fundus_overlays.py --output_dir outputs/demo_fundus --dv_mode max_center_chord
python quick_check_error_maps.py --output_dir outputs/demo_error_polar --input_space polar
python quick_check_error_maps.py --output_dir outputs/demo_error_cartesian --input_space cartesian
```

These demonstrations check readouts, coordinates, and rendering. They do not assess model performance or clinical validity.

