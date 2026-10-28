# Dv 两种计算方法：手动合入包

本包保留原固定轴定义 `fixed_vertical`，新增 `max_center_chord`，由 config 选择。
默认值仍为 `fixed_vertical`，未配置新字段时延续旧行为。

## 包内内容与适用基线

- `updated_files/`：本次全部 10 个改动/新增文件的完整版本，目录结构与项目一致。
- `dv_modes.patch`：只包含本次 Dv 功能的增量补丁。
- `changed_files.json`：逐文件的基线与新文件 SHA-256。
- `validation.json`：验证结果、运行环境及未覆盖范围。
- `updated_files/docs/DV_MODES.md`：两种定义、单位、hard/soft 及记录协议的详细说明。

补丁基于此前提供的 `medseg_research.zip`，不重复包含 mask 导出、依赖或其他功能修改。
如果你已手动修改过文件，请按下列函数/位置合并；不要直接覆盖自己的其他改动。

## 必须合入的 3 个主要文件

| 文件 | 需要合入的位置 |
|---|---|
| `configs/quantification/pmw_pvdr.yaml` | 新增 `dv_mode`，注明固定轴角度的适用模式 |
| `src/quantification/polar_readout.py` | helper、构造参数与校验、共享 `_summarize`、builder 参数、metadata |
| `src/evaluation/quantification_records.py` | 新协议常量及 `quantification_protocol()` |

### 1. configs/quantification/pmw_pvdr.yaml

在现有 `vertical_angle_deg` 附近加入：

```yaml
# fixed_vertical | max_center_chord
dv_mode: fixed_vertical

# Only used by fixed_vertical.
vertical_angle_deg: 90.0
```

此配置组文件已有 `# @package quantification`，这里不再套一层 `quantification:`。
需要新方式时改为 `dv_mode: max_center_chord`。
也可以在原有训练/评估命令中覆盖 `quantification.dv_mode=max_center_chord`。
如果尚未选择该配置组，先添加 `+quantification=pmw_pvdr`；已有配置组时无需再次添加。

### 2. src/quantification/polar_readout.py

#### 2.1 保留 `_opposite_vertical()`，在它后面新增以下函数

```python
def _max_center_chord(profile: Tensor) -> Tensor:
    """Longest opposite-ray sum through the polar origin.

    ``profile[..., i]`` is the OD radius at angle ``2*pi*i/A``. Even angular
    grids have an exact opposite row. On odd grids, the opposite direction
    lies halfway between two rows and uses periodic linear interpolation.
    Maximize the paired sum, rather than choosing the longest single radius.
    """
    if profile.ndim < 1 or profile.shape[-1] == 0:
        raise ValueError("OD profile must have a nonempty angle dimension.")
    angles = profile.shape[-1]
    half = angles // 2
    opposite = torch.roll(profile, shifts=-half, dims=-1)
    if angles % 2:
        opposite = 0.5 * (
            opposite + torch.roll(profile, shifts=-(half + 1), dims=-1)
        )
    return (profile + opposite).amax(dim=-1)
```

该函数输入是 `[batch, angle_bins]` 的半径 profile，最后一维表示角度，已经是长度单位。
因此在这里不再乘 `dr`。偶数角度行直接配对；奇数行的对向角度使用周期线性插值。

#### 2.2 在 `PolarReadout.__init__()` 参数列表末尾追加

```python
dv_mode: str = "fixed_vertical",
```

在构造函数现有参数校验区加入：

```python
self.dv_mode = str(dv_mode).strip().lower()
if self.dv_mode not in {"fixed_vertical", "max_center_chord"}:
    raise ValueError(
        "quantification.dv_mode must be 'fixed_vertical' or "
        f"'max_center_chord', got {dv_mode!r}."
    )
```

保留原 `self.vertical_angle_deg` 字段及校验。它只决定固定轴模式的方向。

#### 2.3 在 `_summarize()` 内替换 Dv 的一行计算

原来：

```python
dv = _opposite_vertical(od, self.vertical_angle_deg)
```

替换为：

```python
if self.dv_mode == "max_center_chord":
    dv = _max_center_chord(od)
else:
    dv = _opposite_vertical(od, self.vertical_angle_deg)
```

保留周围的 `pmw = width.amax(dim=-1)`、返回字典和 PVDR 分母保护。
现有 `pvdr = pmw / dv.clamp_min(self.denominator_floor_px)` 自然使用所选 Dv。

#### 2.4 在 `_build_polar_readout()` 的 `readout_type(...)` 调用中追加

```python
dv_mode=q.get("dv_mode", "fixed_vertical"),
```

这样 `build_polar_readout()` 和量化 loss 的 builder 都能使用配置。
旧配置没有此字段时仍回退到固定轴。

#### 2.5 在该 builder 的 `readout.geometry_metadata = {...}` 后、`return readout` 前加入

```python
if readout.dv_mode == "max_center_chord":
    readout.geometry_metadata.pop("vertical_angle_deg", None)
    readout.geometry_metadata.update({
        "dv_mode": readout.dv_mode,
        "diameter_definition": "max_theta[r(theta)+r(theta+pi)]",
        "opposite_angle_deg": 180.0,
        "opposite_interpolation": "periodic_linear",
    })
```

### 3. src/evaluation/quantification_records.py

在原协议常量附近加入一行，旧常量值不变：

```python
QUANTIFICATION_SCHEMA_VERSION = "quantification-v1"
QUANTIFICATION_PROTOCOL = "polar-radial-v1"
MAX_CENTER_CHORD_PROTOCOL = "polar-radial-v2"  # added
```

将现有 `quantification_protocol()` 替换为以下完整版本。
此文件已有 `hashlib`、`json` 及 `_python_values()`，无需新增依赖。

```python
def quantification_protocol(readout) -> dict:
    """Identify the sampling grid and the diameter definition actually used.

    Fixed-axis readouts retain the original v1 geometry encoding so their CSVs
    remain comparable with existing runs. The maximum centre chord is a separate
    v2 protocol; its geometry excludes the unused fixed-axis angle, including any
    stale copy in ``geometry_metadata``.
    """
    dv_mode = str(getattr(readout, "dv_mode", "fixed_vertical")).strip().lower()
    if dv_mode not in {"fixed_vertical", "max_center_chord"}:
        raise ValueError(f"Unsupported quantification dv_mode: {dv_mode!r}")
    geometry = dict(getattr(readout, "geometry_metadata", {}))
    # These descriptors did not exist in v1. Resolve them from the actual mode,
    # rather than trusting copied metadata or changing a fixed-axis run's hash.
    for key in ("dv_mode", "diameter_definition", "opposite_interpolation", "opposite_angle_deg"):
        geometry.pop(key, None)
    geometry.update({
        "angle_bins": int(readout.angle_bins),
        "radius_bins": int(readout.radius_bins),
        "max_radius": float(readout.max_radius),
        "od_class": int(readout.od_class),
        "ppa_class": int(readout.ppa_class),
        "denominator_floor_px": float(readout.denominator_floor_px),
        "mode": "linear", "radial_step": "max_radius/radius_bins",
        "origin_weight": 0, "length_unit": "resized_roi_px",
    })
    if dv_mode == "max_center_chord":
        geometry.pop("vertical_angle_deg", None)
        geometry.pop("vertical_angle", None)
        geometry.update({
            "dv_mode": dv_mode,
            "diameter_definition": "max_theta[r(theta)+r(theta+pi)]",
            "opposite_angle_deg": 180.0,
            "opposite_interpolation": "periodic_linear",
        })
        protocol_name = MAX_CENTER_CHORD_PROTOCOL
    else:
        geometry["vertical_angle_deg"] = float(readout.vertical_angle_deg)
        protocol_name = QUANTIFICATION_PROTOCOL
    grid = _python_values(readout.valid_grid)
    geometry["valid_grid_sha256"] = hashlib.sha256(
        bytes(int(value) for row in grid for value in row)
    ).hexdigest()
    encoded = json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "quant_schema_version": QUANTIFICATION_SCHEMA_VERSION,
        "quant_protocol": protocol_name,
        "quant_geometry_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "quant_geometry_json": encoded,
        "quant_space": "resized_roi_polar",
        "quant_length_unit": "resized_roi_px",
    }
```

固定轴保留 v1 以及旧几何 hash；最大中心连线使用 v2，且不把无效的固定轴角度纳入 hash。
原 `quantification_stats.py` 已检查 protocol/hash，不需要追加统计逻辑。
两种 Dv 定义需要分别汇总，模型比较时用相同模式重新评估双方。

## 其他改动：用于说明和验证

| 文件 | 改动 | 运行新功能是否必须 |
|---|---|---|
| `src/quantification/hard_readout.py` | 注释由“vertical diameter”改为“selected diameter”，计算未变 | 否 |
| `quick_check_quantification.py` | 保留 fixed 检查并新增 max 模式、PVDR 和梯度检查 | 否，建议加入 |
| `tests/test_quantification_readouts.py` | 两模式、配置传递、旧数值和梯度回归 | 否，建议加入 |
| `tests/test_quantification_pipeline.py` | 两模式的 SegModule、病例记录和协议集成 | 否，建议加入 |
| `tests/test_max_center_chord.py` | 新增奇偶角度、对向和最大值、梯度、有效性测试 | 否，建议加入 |
| `tests/test_dv_protocol.py` | 新增旧 hash 兼容及跨定义混用检查 | 否，建议加入 |
| `docs/DV_MODES.md` | 新增完整定义说明 | 否，建议加入 |

`src/quantification/soft_readout.py`、`src/losses/polar_quantification.py`、
`src/evaluation/quantification_stats.py` 和冻结旧实现 fixture 保留原样。
hard、soft 与 loss 共用 `_summarize()`，不需要分别复制新公式。
此补丁面向当前 quantification 公共路径，未改动其他历史独立推理脚本。

## 定义与实际结果示例

固定轴：`Dv = r_OD(theta0) + r_OD(theta0 + pi)`，默认 theta0=90°。
最大中心连线：`Dv = max_theta [r_OD(theta) + r_OD(theta + pi)]`。
这里的 center 就是现有 Polar 原点（图像左侧 r=0），不重新估算中心。
应当先对每组相反方向求和再取最大值，不能先选最长单边半径。

例如 8 个方向的半径为 `[15, 9, 6, 8, 13, 10, 12, 7]`、PMW=5：

| 模式 | Dv | PVDR |
|---|---:|---:|
| fixed_vertical (90°/270°) | 6 + 12 = 18 | 5/18 = 0.277778 |
| max_center_chord | 15 + 13 = 28 | 5/28 = 0.178571 |

hard 用 OD 边界半径；soft 用可微的径向概率占据长度，并应用同样的聚合方法。
两者并非对任意概率输入都数值相等。单位仍是 resize 后 Cartesian ROI 的像素，
与导出的 mask 是否恢复原图大小无关。

## 已完成验证与本地验证命令

已在真实 PyTorch 2.12.1 CPU 环境通过 **111 个 pytest 用例和 16 个统计子测试**，
全部 0 失败、0 跳过；含 autograd、旧实现数值/梯度回归和真实 SegModule 小模型集成。
`quick_check_quantification.py` 也已执行，两种模式均有有限、非零梯度。
补丁已通过 `git apply --check`，实际应用后 10 个文件均与包内最终文件逐字节一致。

本地验证：

```bash
python -m pytest tests/test_max_center_chord.py tests/test_dv_protocol.py \
  tests/test_quantification_readouts.py tests/test_quantification_pipeline.py \
  tests/test_quantification_statistics.py -q
python quick_check_quantification.py
```

如果要自动合入，在项目根目录运行（路径替换为你解压后的实际位置）：

```bash
git apply --check /path/to/dv_modes_manual/dv_modes.patch
git apply /path/to/dv_modes_manual/dv_modes.patch
```

如果检查提示上下文不匹配，请按上面的函数位置手动合并。
本次未执行你的真实数据集训练、用户 checkpoint 评估、CUDA/AMP 或 DDP 测试。
完整数值与几何说明见 `updated_files/docs/DV_MODES.md`。
