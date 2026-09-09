# simesh 当前应用手册

本文面向使用当前根目录版本进行物理分析的用户。按任务选择接口即可，
不需要先阅读核心开发过程。完整签名见[接口参考](api-reference.md)，
接口复杂度和资源开销的检查结果见[应用接口审查](interface-review.md)。

## 1. 安装与最短工作流

在仓库根目录执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

日常分析只需三个步骤：打开输入、取得需要的字段、调用计算函数。
做体积分等内部单元统计用 `read_fields`；需要插值、导数或积分路径时用 `prepare`。

```python
import numpy as np
import simesh as sm
from simesh import applications as app

limit = 2 * 1024**3
with sm.open_amrvac("snapshot.dat", memory_limit=limit) as source:
    metadata = source.metadata
    magnetic = sm.prepare(source, ("b1", "b2", "b3"),
                          scheme="coordinate-phase", memory_limit=limit)

lo, hi = magnetic.mesh.lower, magnetic.mesh.upper
width = hi - lo
points = sm.PointSet(np.array([lo + .5 * width]))
sampled = app.sample(magnetic, points, components="b3")
print(sampled.values, sampled.usable)
```

文件关闭后，`magnetic` 仍可独立使用。这个例子只准备三个磁场分量，不会为了
采样 `b3` 再复制一份全域单分量场。字段名应与实际文件一致。
后续各节展示不同任务，按需执行；不必同时计算和保留全部中间场。

`coordinate-phase` 用于全域准备；盒选区和有限容量准备使用 `exact-phase`：

```python
with sm.open_amrvac("snapshot.dat") as source:
    regional = sm.prepare(source, ("b1", "b2", "b3"),
                          region=(lo + .2 * width, lo + .8 * width),
                          scheme="exact-phase", memory_limit=limit)
```

区域选择覆盖所有相交的完整叶块，并从原网格读取必要支撑。区域边缘不成为物理边界。
两个准备方案的数值定义不同，不应在比较结果时随意互换。

## 2. 数据范围与最少需要理解的对象

| 对象 | 用途 | 生命周期与代价 |
| --- | --- | --- |
| `Source` | 文件或数组输入及字段目录 | 用完关闭；读取和准备会生成独立字段 |
| `Fields` | 可重复消费的原生 AMR 字段 | 含字段定义、叶块覆盖和有效支撑；不是均匀网格 |
| `PointSet`、`RaySet` | 需要编号、图像布局和保存恢复的点/射线 | 拥有几何数组；重复分析时复用同一对象 |
| 应用结果 | 图、射线、磁力线和剖面 | 保存几何、编号、值或状态；大结果按批消费 |

- 原生文件入口支持非周期 Cartesian 3D AMRVAC v5 普通字段，网格需满足支持的 AMR 平衡条件。
  当前原生准备采用连续物理边界；不提供周期、球坐标、CT 面变量或 GPU 分析。
- `simesh.amrvac` 保留可变 Dataset、Cartesian 2D singleton-z、普通读写和既有 VTK。
  二维 Dataset 不能直接转换成原生三维分析输入。
- 数值使用 float64。数组输入是 `(leaf, component, x, y, z)`；
  `Fields.values` 是 `(slot, x, y, z, component)`。
  使用 `interior()` 或 `window()` 访问内部值，不要把鬼点算进体积分。
- `read_fields` 没有有效 halo；`prepare` 完成两层。插值至少需要一层；
  一阶差分消耗一层，自动 curl 后再积分 twist 通常需要输入有两层。
  分配了内存不等于支撑已经有效。
- 元数据中的时间、参数和单位标签不自动建立物理归一化或选择能量模型。

## 3. 按目标选择入口

| 目标 | 推荐入口 | 输出 |
| --- | --- | --- |
| 少量点采样 | `sm.sample`；需要编号/保存时用 `app.sample` | 数组元组；或 `SampledPoints` |
| 切片、场图 | `app.field_map` | 带平面和编号的采样结果 |
| 均匀体数据 | `app.uniform_grid`；大体积用 `sm.iter_uniform` | 完整体数据；或独立切片 |
| 电流、磁压、磁能等 | `sm.current_density`、`magnetic_pressure`、`magnetic_energy_density` | 可继续消费的字段 |
| 梯度、散度、旋度和公式 | `sm.gradient`、`divergence`、`curl`、`derive_many` | 可继续消费的字段 |
| Q/twist 面诊断 | `app.surface_diagnostics`、`bottom_diagnostics` | 可筛选和保存的 `ConnectivityMap` |
| 保存完整曲线 | `app.trace`；大批量用 `app.iter_lines` | `LineSet` 或其批次 |
| 沿线温度、密度、电流 | `sm.sample_line_profiles` | `LineProfiles` |
| 速度、热压、温度、beta 等 | `sm.mhd_fields` | 显式理想 MHD 模型恢复的字段 |
| 体积分、统计、矩形面通量 | `sm.volume_integral` 等 | 值、单位与覆盖信息 |
| 标量/热 LOS | `app.los`、`app.thermal_los` | 带射线几何和状态的 `RayResult` |
| 保存应用结果 | `sm.save_result`、`load_result` | 带版本的 NPZ 文件 |
| 普通 AMRVAC 数据产品 | `sm.write_amrvac` 或 `simesh.amrvac` 写接口 | 普通 `.dat` 文件 |

需要应用结果对象时优先使用 `app`；只要数组、原始追踪摘要或底层执行控制时再使用
对应的 `sm` 接口。不要仅因函数都公开，就在一个流程里重复调用两套入口。

## 4. 场图与标准物理量

磁场和坐标的换算系数由调用者明确提供。以下数值仅展示配置方法，必须匹配实际数据。

```python
magnetic_units = sm.MagneticUnits(field_tesla=1e-3, length_m=1e6)
current = sm.current_density(magnetic, units=magnetic_units)
current_strength = sm.magnitude(current)
energy = sm.magnetic_energy_density(magnetic, units=magnetic_units)
div_b = sm.divergence(magnetic)

plane = sm.Plane(lo + np.array([0., 0., .5]) * width,
                 [width[0], 0., 0.], [0., width[1], 0.], (64, 64))
current_map = app.field_map(current_strength, plane)
image = np.where(current_map.usable.reshape(plane.shape),
                 current_map.image[..., 0], np.nan)
```

电流采用 `curl(B)/mu`，返回 A/m²；磁压与磁能密度采用 `B²/(2mu)`，
分别标记 Pa 和 J/m³。磁导率为均匀标量，默认值为通常采用的 `4π×10⁻⁷ H/m`。
`current_density` 直接把换算系数放入差分计算，不先生成完整 curl 再缩放。
QSL 的 `curl_field` 则必须是相应磁场的原始 `sm.curl(magnetic)`，不能传物理电流。

场图的 `valid` 只描述采样覆盖；`usable` 还排除非有限值。`image` 只是按布局重排，
不会把无效结果自动补成有效数据。

## 5. Q/twist、筛选和磁力线

```python
diagnostic = app.bottom_diagnostics(
    magnetic, (32, 32), quantities=("q", "twist"),
    step_fraction=.125, max_steps=4000,
)
log_q = np.where(diagnostic.image("q_valid"),
                 diagnostic.image("log10_q"), np.nan)
selected = diagnostic.threshold(q_min=1e4, abs_twist_min=1., mode="any")
trace_step = float(magnetic.mesh.spacing.min()) * .125
lines = app.trace(magnetic, selected, step=trace_step, max_steps=2000)
```

阈值仅为示例；筛选结果可以为空。`quantities=("q",)` 跳过 twist，
`quantities=("twist",)` 跳过 Q 的邻线/梯度计算。默认联合请求会计算二者。
Q/twist 阶段不保存路径，选中后再单独追踪。

- QSL 给出 Q、Q⊥、对数值、脚点、长度、状态和可选 twist。默认用相邻种子脚点差分，
  `method="variational"` 使用单位方向场的导数传播横向变化；后者需要两层输入 halo。
- `normalization="mapping"` 从映射面积计算分母；`"flux"` 使用磁通关系。
  离散场不满足相应假设时二者可以不同，不能静默替换。
- `delta` 是邻线扰动的物理坐标距离；`step_fraction` 限制局部积分步长。
  新工况需分别检查步长、扰动间距和网格分辨率的收敛。
- QSL 定位目标边界脚点；`app.trace` 保存最后接受的路径前缀，端点不保证与 QSL 脚点相同。
- `LineSet` 默认含正反两支，`branch(seed_id, -1/+1)` 返回一支，`line(seed_id)` 拼接显示曲线。
  `offsets` 长度为 `2*n+1`，`termination` 形状为 `(n, 2)`。
- 检查 `q_valid`、`twist_valid` 和终止状态。有限 twist 也可能只来自未完成的路径前缀。

已有任意种子坐标时用 `app.connectivity(magnetic, point_set, quantities=...)`；
只需原始数组结果时用 `sm.qsl(magnetic, positions, twist=False/True)`。

## 6. MHD 恢复

MHD 接口要求经典 `m=rho*v`、恒定 gamma、完全电离 H/He，以及明确的
`energy_kind="total"` 或 `"internal"`。背景场分裂、其他能量库和不同 EOS
不能只凭字段名 `e` 当作当前模型。

可以把守恒量与磁场分组准备，避免先准备八分量大场，再复制出三分量磁场供追踪。
两个组必须来自同一个 `Mesh`；在同一个 Source 内准备即可。

```python
model = sm.IdealMHD(
    gamma=5/3, energy_kind="total", composition=sm.CoronalComposition(),
    units=sm.MHDUnits(density_kg_m3=1e-12, momentum_kg_m2_s=1e-8,
                      energy_j_m3=1., magnetic=magnetic_units),
)
with sm.open_amrvac("snapshot.dat") as source:
    magnetic = sm.prepare(source, ("b1", "b2", "b3"), scheme="coordinate-phase")
    conserved = sm.prepare(source, ("rho", "m1", "m2", "m3", "e"),
                           scheme="coordinate-phase")
state = sm.mhd_fields(conserved, magnetic=magnetic, model=model,
                      outputs=("density", "temperature"))
```

返回的密度为 kg/m³，温度为 K，其他有量纲输出使用 SI。
可选输出包括速度、速率、内能密度、热压、beta、声速、Alfvén 速度、两种马赫数和状态。
`invalid="raise"` 默认拒绝非法状态；`"nan"` 保留状态并将无效物理值标记 NaN。
状态列是分类数据，不能插值。

`outputs` 同时选择输出列和可选诊断。密度、能量、热压、温度等完整物理状态检查始终执行；
只请求速度时会跳过 beta、声速、Alfvén 速度和马赫数。请求马赫数会计算所需的速度诊断；
请求 `status` 时检查全部诊断，默认输出包含 `status`。

`preparation_stats["evaluated_diagnostics"]` 列出本次计算的可选诊断，含依赖。
`status_counts` 的 `UNREPRESENTABLE_DIAGNOSTIC` 只统计该范围；未检查任何诊断时为
`None`，不表示全部通过。物理非法状态计数和零磁场标记始终检查。
同一次调用尽量取得本次任务确实需要的恢复量，避免对同一组输入多次恢复。
单位配置也不会根据文件头自动推断。

## 7. 积分、统计与面通量

这些运算直接使用内部单元值和真实 AMR 单元体积，不需要鬼点或先采样成均匀网格。

```python
with sm.open_amrvac("snapshot.dat") as source:
    raw_density = sm.read_fields(source, "rho")
rho_si = sm.derive(raw_density, "density", lambda ctx: ctx.field("rho") * 1e-12,
                   units="kg m^-3")
lengths = sm.LengthUnits(scale=1e6, unit="m")
mass = sm.volume_integral(rho_si, "density", units=lengths)
mean = sm.weighted_mean(rho_si, "density", units=lengths)
limits = sm.extrema(rho_si, "density")
print(mass.value, mass.units, mass.coverage.complete)
```

密度和长度系数仍须匹配真实输入。`weighted_mean` 未提供权重时为体积平均；
质量加权温度需要把物理密度组作为 `weights`，不能把单元数平均当成质量平均。

```python
mean_temperature = sm.weighted_mean(
    state, "temperature", weights=state, weight_component="density", units=lengths,
)
temperature_histogram = sm.histogram(state, [1e5, 1e6, 1e7], "temperature", units=lengths)
bz_tesla = sm.derive(magnetic, "bz_T", lambda ctx: ctx.field("b3") * magnetic_units.field_tesla,
                     units="T")
bottom = sm.AxisAlignedSurface("z", float(lo[2]), [lo[:2], hi[:2]], normal=-1)
flux = sm.surface_flux(bz_tesla, bottom, "bz_T", units=lengths)
```

矩形选区按单元交叠体积/面积加权，采用所提供内部值的分片常数表示。
面通量明确选择法向分量；这是单元中心场的单侧取值，不是 CT 面通量，
也不保证粗细网格两侧通量一致。`side` 指定内部面从哪一侧取值。

默认 `missing="raise"`、`nonfinite="raise"`；用 `"omit"` 时结果只描述有效覆盖，
必须同时保留 `coverage`，不要将其当成完整选区总量。

## 8. 沿线物理量

`sample_line_profiles` 使用已有曲线坐标，不重新追踪。下面假定 `state` 与 `lines`
来自同一物理快照、坐标系和单位约定；从文件恢复的曲线不能自动证明这一点。

```python
profiles = sm.sample_line_profiles(
    state, lines, components=("temperature", "density"),
    length_units=lengths, point_batch=4096,
)
if len(profiles.seed_ids):
    branch = profiles.branch(int(profiles.seed_ids[0]), +1)
    distance = branch.arclength
    temperature = np.where(branch.usable[:, 0], branch.values[:, 0], np.nan)
```

每支弧长从种子重新计零；拼接显示时负支用负弧长，两份种子样本仍保留。
弧长来自已存折线，不是流体运动时间。位置保持原坐标，`length_units` 只转换弧长。
默认 `boundary="interior"` 只将恰好处于全域上边界的采样坐标移到最近内部可表示值，
不修改存储曲线，并用 `boundary_adjusted` 标记。`valid` 表示覆盖，`finite`、`usable`
按分量记录有限性；采样有效不等于磁力线积分已完成。

## 9. 标量与热 LOS

```python
with sm.open_amrvac("snapshot.dat") as source:
    density = sm.prepare(source, "rho", scheme="coordinate-phase")
direction = (.3, .2, 1.)
camera = sm.orthographic_plane(lo, hi, direction, (64, 64))
rays = sm.RaySet.from_plane(camera, direction)
column = app.los(density, rays, component=0)
column_image = np.where(column.valid.reshape(camera.shape), column.image, np.nan)

thermal = sm.thermal_fields(density, 1e6, density_unit_g_cm3=1e-15,
                            temperature_label="isothermal 1 MK example")
emission = app.thermal_los(thermal, rays, length_unit_cm=1e8)
```

标量 LOS 的单位是所选字段单位乘坐标长度。`app.los` 的 `component` 当前要求整数，
与采样函数接受字段名的规则不同。截断深度放在 `RaySet.near/far`，方向在 `directions`。

热 LOS 保留历史 AIA171 响应，需要显式密度、长度和温度约定。
温度也可来自 MHD 恢复：

```python
thermal_state = sm.thermal_fields(
    state, state, density_component=0, temperature_component=1,
    density_unit_g_cm3=1e-3, temperature_label="recovered ideal MHD temperature",
)
recovered_emission = app.thermal_los(thermal_state, rays, length_unit_cm=1e8)
```

这里 `state` 的列顺序为先密度、后温度。默认先插值热力学量，再计算非线性响应；
`order="emissivity-first"` 先计算节点发射率，属于不同重构，不能作为单纯提速开关。
`subdivisions` 控制非线性求积细分，仍需按工况检查收敛。`complete` 表示所有射线有效，
`valid/status` 区分完整、空射线、缺覆盖和采样上限等情况。

## 10. 结果保存、恢复与 `.dat` 输出

```python
from pathlib import Path

output = Path("application-output")
output.mkdir(exist_ok=True)
sm.save_result(output / "current.result.npz", current_map,
               metadata={"field_tesla": 1e-3, "length_m": 1e6})
sm.save_result(output / "connectivity.result.npz", diagnostic,
               metadata={"step_fraction": .125, "max_steps": 4000})
sm.save_result(output / "lines.result.npz", lines,
               metadata={"step": trace_step, "max_steps": 2000})
sm.save_result(output / "profiles.result.npz", profiles)
sm.save_result(output / "thermal.result.npz", emission)
loaded = sm.load_result(output / "profiles.result.npz")
restored_profiles = loaded.result
```

默认拒绝覆盖，重跑时显式传 `overwrite=True` 或换输出目录。
文件保存支持的类型、数组、几何、单位与状态；只有结果中已有的计算参数才自动保存。
其余步长、准备方案、阈值、物理模型和坐标单位需放入 `metadata`。
`metadata.to_dict()` 可记录先前保留的文件头；`source={...}` 只是调用者提供的来源说明。
读取不验证原快照，也不恢复内存中的来源身份凭据。

| 数据产品 | 写入 | 读取与限制 |
| --- | --- | --- |
| `PointSet`、`RaySet`、`SampledPoints`、`ConnectivityMap`、`QSLResult`、`RayResult`、`LineSet`、`LineProfiles`、`UniformResult` | `save_result` | `load_result(...).result` 恢复对应对象 |
| 积分、极值、直方图 | 显式 JSON/NumPy，或支持结果的 `metadata` | 不直接恢复这些归约类；记录单位、覆盖和直方图尾部 |
| 完整原生字段 | `write_amrvac` | 完整原网格覆盖、匹配的文件头；普通内部值，不含 halo 或 CT |
| 旧式 Dataset/均匀数组 | `simesh.amrvac` 写函数 | 保留普通 `.dat` 与二维约定 |
| level-1 VTK | `simesh.amrvac.datfile_to_vtk` | 既有结构化点格式与端点坐标约定；不是磁力线 PolyData |
| 示例自定义 `products.npz` | `numpy.savez_compressed` | `numpy.load(..., allow_pickle=False)`；不能传给 `load_result` |

原始 `Fields`、`TraceResult`、`SliceResult`、`LOSResult`、`ThermalLOSResult`
不直接接受 `save_result`。需要保存应用对象时使用前面的 `app` 调用；
不要假设任意结果都能使用同一个保存入口。

```python
sm.write_amrvac(output / "magnetic.dat", magnetic, metadata=metadata)
```

`.dat` 不编码字段单位、准备方案或派生来源，也不保证导出的分析场可用于模拟重启。
区域字段不能直接写成完整原始森林。应另存所需的物理解释。

## 11. 大输出与开销控制

| 情况 | 选择 |
| --- | --- |
| 只做内部积分 | `read_fields`，避免不需要的 halo |
| 已有大字段，只采样少量列 | 在 `sample`、`field_map`、`uniform_grid`、剖面函数中传 `components` |
| 只计算 Q 或 twist | 明确 `quantities`，避免默认联合计算 |
| 保存大量轨迹 | `app.iter_lines`，控制 `seed_batch` 和必要的 `max_steps` |
| 大均匀体数据 | `sm.iter_uniform` 逐切片写出；或提供 `output` 数组/memmap |
| 不需要完整场组的多输出公式 | `derive_many` 直接选输入组；避免为了单次消费反复 `select_fields`/`merge_fields` |

`select_fields`、`merge_fields` 都会复制数值，目的是获得独立、紧凑的字段组；
它们不是轻量视图。`PointSet` 拷贝坐标并分配 ID，适合建立一次后复用。
若只需计算已有组合场的电流，直接传 `current_density(..., components=("b1","b2","b3"))`。
需要将多组场的少量列组合并换算单位时，`derive_many` 可以按组读取并直接写出所需列，
省去先选场、转换整组、再合并的中间数组，见
[恢复状态示例](../examples/recovered_state_analysis.py)。
`app.trace` 和 `app.iter_lines` 使用固定大小的路径段缓冲，只保留实际接受的点，
最后打包一次。`max_steps` 控制积分上限，不决定初始路径容量；缓冲段满不会终止积分。
`app.trace` 仍需容纳完整结果，`app.iter_lines` 只汇总当前种子分片。
原生 `sm.trace(..., trajectories=True)` 及 `sm.iter_traces` 仍返回稠密数组，
其路径容量仍随 `max_steps` 增长。

```python
line_batches = app.iter_lines(magnetic, selected, seed_batch=64,
                              step=trace_step, max_steps=2000)
profile_batches = sm.iter_line_profiles(state, line_batches,
                                        components=("temperature", "density"),
                                        length_units=lengths)
sm.save_result_shards(output / "profile-shards", profile_batches,
                      seed_ids=selected.ids)
shards = sm.open_result_shards(output / "profile-shards")
```

分片目录必须不存在；通过 `shards.load(index).result` 读取一个分片。
清单完成只代表全部种子交付完毕，不代表全部积分成功。分片也不是积分状态断点续算。
新写入使用版本 2：`seed_ids.npy` 保存完整编号一次，每片有独立的 `index-*.json`，
结束时原子发布完整 `manifest.json`。累计元数据写入量随种子数和分片数近似线性增长。
中断时，读取器从已发布的索引恢复完成分片；没有索引的残留 NPZ 不算已交付，
`complete` 仍为假。版本 1 文件继续可读。通过 `shards.seed_ids` 获取两个版本的编号，
不要依赖清单内部的编号表示。极多小分片仍有文件管理成本，详见[资源审查](interface-review.md)。

`memory_limit` 是单次操作计入的受控数组预算，不是进程 RSS 上限，也不自动
计入调用者另外保留的所有结果。NPZ 保存仍为可写输出和可能有可写别名的数组建立快照；
由 `PointSet`、`RaySet`、`Plane` 构造器独立持有的只读几何可在保存期间直接使用。
仅有只读标记的 `LineSet` 或结果数组仍会快照，避免遗漏外部可写别名。
保存期间不要替换几何或改变其写保护；快照捕获时也不支持并发修改输入。
压缩输入按固定小块写入，但整结果快照和结构检查仍需空间，不能把它视为固定内存的保存接口。
QSL 和热 LOS 分批输出时，输入字段仍需驻留内存。`bounded` 提供部分消费者的
显式有限容量输入路径，但不是所有接口的自动低内存模式。

## 12. 可直接运行的示例

```bash
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-standard
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-quantitative
```

- 标准示例生成自定义 `products.npz` 和 `summary.json`，包含磁场诊断、选线、
  标量/热 LOS 与状态。安装 `.[plot]` 后，可加 `--plot` 或对已有文件执行 `--render-only`，
  生成 `magnetic-applications.png`、`los-applications.png`。
- 定量示例生成 `thermal.result.npz` 和 `streamlines.result.npz`，使用 `load_result` 读取。
  它还计算并检查沿线剖面，但不自动保存剖面文件。
- 定量例子的已知结果为质量 `2.4e-12 kg`、质量加权温度 `840000 K`、
  底面向外磁通 `-2e-4 T*m²`。示例是解析场验证，不是实际模拟的单位标定或极端 QSL 验收。

独立势场和解析磁场构型位于 `simesh.tools`：
`potential_field_green` 接收底面磁场数组，`tools.configurations` 提供双极场、
偶极场、单极场、RBSL、TDm 和 fan 构型。构型坐标通常以第一维表示三个分量，
与点采样接口的 `(n,3)` 布局不同，调用前应核对[接口签名](api-reference.md)。
