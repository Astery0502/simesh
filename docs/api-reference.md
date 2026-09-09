# 当前应用接口参考

本页依据当前运行源码核对签名。应用选择与示例见[中文应用手册](application-guide.md)。
以下签名用于查阅，并非要求一次调用所有入口；高级内存/执行选项可保持默认值。
省略内部 Source 构造器、Fields 存储构造器以及编译内核接口。

```python
import simesh as sm
from simesh import applications as app, amrvac, tools
from simesh.tools import configurations
```

## 参数和结果中容易混淆的地方

| 接口 | 需要注意 |
| --- | --- |
| `read_fields`、`prepare` | 字段名最直观；Source 整数选择器表示其当前字段目录，受限 Source 的本地编号不是完整文件头编号 |
| `sample`、`field_map`、`select_fields`、剖面 | `components` 可用名称或局部分量整数 |
| `curl` | `components` 当前要求三个整数；要按名字写差分可用 `derivative`，或先准备恰好三列磁场 |
| `derivative` | 每个输出对应一组 `(字段名或编号, 方向名或编号, 系数)`，`definitions` 描述各输出 |
| `app.los` | `component` 当前只收整数；近远深度由 `RaySet` 给出 |
| `sm.trace` | 原始 `TraceResult`；默认方向 +1、不存轨迹、不算 twist |
| `app.trace` | 紧凑 `LineSet`；默认双向并存轨迹，方向用字符串；不接收原生 `twist` 参数 |
| `app.surface_diagnostics`、`bottom_diagnostics` | 默认同时算 Q 和 twist，可用 `quantities` 只选一种 |
| `sm.qsl` | 总会算 Q，`twist=True` 是默认值；`method`、`normalization` 和扰动参数改变数值定义 |
| `mhd_fields` | 完整物理状态检查始终执行；可选诊断按 `outputs` 及依赖计算，包含 `status` 时检查全部诊断 |
| `output` | 调用者提供目标数组；不能与输入或其他输出重叠，失败时可能已有部分写入 |
| `memory_limit` | 单次调用计入的数组预算；不是进程 RSS 上限，也不是结果文件大小限制 |

`app.sample`/`field_map` 返回 `SampledPoints`；`uniform_grid` 返回 `UniformResult`；
面诊断返回 `ConnectivityMap`；应用追踪返回 `LineSet`；沿线采样返回 `LineProfiles`；
应用 LOS 返回 `RayResult`。这些类型与编号、几何和保存格式的对应关系见应用手册。
`save_result` 不直接接受原生追踪、切片、LOS、Fields 或归约结果类。

## 1. 输入与准备

```python
sm.open_amrvac(path, *, fields=None, units=None, memory_limit=None)
sm.read_fields(source, fields=None, *, region=None, leaf_ids=None, memory_limit=None)
sm.prepare(source, fields=None, *, region=None, leaf_ids=None, scheme, workers=1, memory_limit=None,
    support_capacity=128, backend='threadpool', plan=None)
sm.source_from_arrays(mesh, values, fields, *, units=None, copy=True, memory_limit=None, metadata=None)
sm.source_from_dataset(dataset, fields=None, *, units=None, memory_limit=None)
sm.iter_prepared(source, fields=None, *, region=None, leaf_ids=None, scheme, batch_size=128,
    support_capacity=128, memory_limit=None)
```

## 2. 几何、采样与均匀网格

```python
sm.Plane(origin: numpy.ndarray, u: numpy.ndarray, v: numpy.ndarray, shape: tuple) -> None
sm.PointSet(positions: numpy.ndarray, ids: numpy.ndarray | None = None, shape: tuple | None = None, normals:
    numpy.ndarray | None = None, plane: simesh.slices.Plane | None = None) -> None
sm.RaySet(origins: simesh.geometry.PointSet, directions: numpy.ndarray, near: object = 0.0, far: object = inf) -> None
sm.sample(fields, points, *, components=None, output=None, workers=1, memory_limit=None)
sm.sample_plane(fields, plane, *, components=None, output=None, tile_rows=64, workers=1, memory_limit=None)
sm.iter_uniform(fields, resolution, *, components=None, bounds=None, tile_rows=64, workers=1, memory_limit=None)
sm.orthographic_plane(lower, upper, direction, shape)
```

## 3. 应用采样与图结果

```python
app.sample(fields, points, *, components=None, output=None, workers=1, memory_limit=None)
app.field_map(fields, surface, *, components=None, output=None, workers=1, memory_limit=None)
app.uniform_grid(fields, resolution, *, components=None, output=None, bounds=None, workers=1, tile_rows=64,
    memory_limit=None)
```

## 4. 数学与磁场诊断

```python
sm.derive(inputs, name, func, *, units='code', memory_limit=None)
sm.derive_many(inputs, definitions, func, *, memory_limit=None)
sm.select_fields(fields, components=None, *, names=None, memory_limit=None)
sm.merge_fields(inputs, *, names=None, memory_limit=None)
sm.derivative(fields, terms, definitions, *, output=None, workers=1, memory_limit=None)
sm.curl(fields, components=(0, 1, 2), *, output=None, workers=1, memory_limit=None)
sm.gradient(fields, component=0, *, name=None, workers=1, memory_limit=None)
sm.divergence(fields, components=(0, 1, 2), *, name='divergence', workers=1, memory_limit=None)
sm.magnitude(fields, components=None, *, name='magnitude', memory_limit=None)
sm.dot(left, right, *, left_components=None, right_components=None, name='dot_product', memory_limit=None)
sm.MagneticUnits(field_tesla: float, length_m: float, permeability_h_m: float = 1.2566370614359173e-06) -> None
sm.current_density(fields, *, units, components=(0, 1, 2), workers=1, memory_limit=None)
sm.magnetic_pressure(fields, *, units, components=(0, 1, 2), memory_limit=None)
sm.magnetic_energy_density(fields, *, units, components=(0, 1, 2), memory_limit=None)
```

## 5. 磁连接性与曲线

```python
app.connectivity(fields, points, *, quantities=None, memory_limit=None, **controls)
app.surface_diagnostics(fields, surface, *, quantities=('q', 'twist'), memory_limit=None, **controls)
app.bottom_diagnostics(fields, shape=(128, 128), *, quantities=('q', 'twist'), ids=None, memory_limit=None, **controls)
app.iter_connectivity(fields, points, *, quantities=None, seed_batch=256, memory_limit=None, **controls)
app.trace(fields, points, *, direction='both', step, max_steps=1000, max_length=inf, null_threshold=0.0,
    workers=1, backend='threadpool', schedule='static', seed_batch=128, memory_limit=None)
app.iter_lines(fields, points, *, seed_batch=128, memory_limit=None, **controls)
```

## 6. 原生诊断、追踪与沿线量

```python
sm.qsl(fields, seeds, *, bounds=None, step_fraction=0.25, step=None, max_steps=10000, max_length=inf,
    null_threshold=0.0, boundary_tolerance=None, local_radius=None, normalization='mapping',
    method='finite-difference', delta=None, twist=True, curl_field=None, workers=1, seed_batch=256,
    memory_limit=None)
sm.line_diagnostics(fields, seeds, *, quantities=('q', 'twist'), memory_limit=None, **controls)
sm.iter_qsl(fields, seeds, *, bounds=None, step_fraction=0.25, step=None, max_steps=10000, max_length=inf,
    null_threshold=0.0, boundary_tolerance=None, local_radius=None, normalization='mapping',
    method='finite-difference', delta=None, twist=True, curl_field=None, workers=1, seed_batch=256,
    memory_limit=None)
sm.trace(fields, seeds, *, seed_ids=None, step, max_steps=1000, max_length=inf, null_threshold=0.0,
    direction=1, workers=1, seed_batch=256, trajectories=False, twist=False, curl_field=None,
    memory_limit=None, backend='threadpool', schedule='static')
sm.iter_traces(fields, seeds, *, seed_ids=None, step, max_steps=1000, max_length=inf, null_threshold=0.0,
    direction=1, workers=1, seed_batch=256, trajectories=False, twist=False, curl_field=None,
    memory_limit=None, backend='threadpool', schedule='static')
sm.retrace(fields, result, selected_seed_ids, **kwargs)
sm.sample_line_profiles(fields, lines, components=None, *, point_batch=4096, workers=1, length_units=None,
    boundary='interior', memory_limit=None)
sm.iter_line_profiles(fields, line_batches, components=None, *, memory_limit=None, **controls)
```

## 7. MHD 与原生统计

```python
sm.MHDUnits(*, density_kg_m3: float, momentum_kg_m2_s: float, energy_j_m3: float, magnetic:
    simesh.diagnostics.MagneticUnits) -> None
sm.IdealMHD(*, gamma: float, energy_kind: str, composition: simesh.physics.thermal.CoronalComposition,
    units: simesh.physics.mhd.MHDUnits) -> None
sm.mhd_fields(conserved, *, model, magnetic=None, density='rho', momentum=('m1', 'm2', 'm3'), energy='e',
    magnetic_components=('b1', 'b2', 'b3'), outputs=('density', 'velocity', 'pressure', 'temperature',
    'beta', 'sound_speed', 'alfven_speed', 'sonic_mach', 'alfven_mach', 'status'), invalid='raise',
    memory_limit=None)
sm.LengthUnits(scale: float, unit: str) -> None
sm.AxisAlignedSurface(axis: int | str, coordinate: float, bounds: numpy.ndarray, normal: int = 1, side: str
    = 'positive') -> None
sm.volume_integral(fields, component=0, *, region=None, units=None, missing='raise', nonfinite='raise')
sm.weighted_mean(fields, component=0, *, weights=None, weight_component=0, weight_mode='density',
    region=None, units=None, missing='raise', nonfinite='raise')
sm.extrema(fields, component=0, *, region=None, units=None, missing='raise', nonfinite='raise')
sm.histogram(fields, edges, component=0, *, weights=None, weight_component=0, weight_mode='density',
    region=None, units=None, missing='raise', nonfinite='raise')
sm.surface_flux(fields, surface, component=0, *, units=None, missing='raise', nonfinite='raise')
```

`mhd_fields` 始终执行完整物理状态检查；可选诊断由 `outputs` 及依赖决定。
包含 `status` 时检查全部诊断。`preparation_stats["evaluated_diagnostics"]` 给出范围；
`status_counts` 中诊断溢出计数只适用于此范围，没有诊断时为 `None`。
`invalid_state_counts` 和物理错误的拒绝/NaN 行为不变。

## 8. 热力学与射线应用

```python
sm.CoronalComposition(helium_abundance: float = 0.1) -> None
sm.AIA171(density_convention: str = 'electron', composition: simesh.physics.thermal.CoronalComposition =
    CoronalComposition(helium_abundance=0.1)) -> None
sm.thermal_fields(density, temperature, *, density_unit_g_cm3, model=AIA171(density_convention='electron',
    composition=CoronalComposition(helium_abundance=0.1)), density_component=0, temperature_component=0,
    temperature_label, memory_limit=None)
sm.emissivity_fields(thermodynamics, *, model=AIA171(density_convention='electron',
    composition=CoronalComposition(helium_abundance=0.1)), memory_limit=None)
```

## 9. 射线积分

```python
app.los(fields, rays, *, component=0, quadrature='gauss2', step_fraction=0.5, max_samples=1000000,
    workers=1, ray_batch=4096, memory_limit=None)
app.thermal_los(thermodynamics, rays, *, length_unit_cm, model=None, order='thermodynamics-first',
    subdivisions=4, max_samples=1000000, workers=1, ray_batch=4096, memory_limit=None)
```

## 10. 保存与恢复

```python
sm.save_result(path, result, *, metadata=None, source=None, overwrite=False)
sm.load_result(path)
sm.save_result_shards(path, batches, *, seed_ids, metadata=None, source=None)
sm.open_result_shards(path)
sm.write_amrvac(path, fields, *, metadata, overwrite=False, memory_limit=None)
```

`save_result` 的 NPZ 协议仍为版本 1。可写输出及可能有可写别名的数组会快照；
构造器自有的只读 `PointSet`/`RaySet`/`Plane` 几何可直接用于写入。
保存期间不得修改这类几何或其写保护状态；快照捕获也不支持并发修改输入。
压缩按固定输入块处理，完整快照和验证仍占内存，接口没有固定内存保证。

`save_result_shards` 返回 `ResultShards`，新写入为版本 2；`open_result_shards`
兼容版本 1、2。`shards.seed_ids` 返回只读的完整编号数组；`complete` 表示是否交付全部
种子，`len(shards)` 是已提交分片数，`load(index)` 校验并读取单片。
版本 2 的 `manifest["seed_ids"]` 是文件描述对象，版本 1 为编号列表；
应用应使用 `seed_ids` 属性，避免依赖版本相关结构。

版本 2 目录包含 `seed_ids.npy`、`shard-*.npz`、`index-*.json` 和 `manifest.json`。
编号描述保存文件名、数量和 SHA-256；每片索引保存文件名、种子范围、结果类型和 SHA-256。
开始时清单标记未完成，每片先发布 NPZ 再原子发布索引，最后原子发布包含全部索引的完成清单。
未完成目录按连续索引恢复；索引缺失或损坏会报错，未提交的 NPZ 残留不会成为交付结果。
这提供应用中断后的结果检查，不承诺断电持久性，也不提供积分断点续算。

## 11. 保留的 Dataset 与文件接口

```python
amrvac.open_dataset(path: str, *, ghost_width: int = 0, boundary_conditions=None) ->
    simesh.amrvac.amrvac_dataset.AMRVACDataSet
amrvac.read_blocks(path: str, *, field_indices: list[int] | None = None, ghost_width: int = 0,
    include_ghosts: bool = False, boundary_conditions=None) -> numpy.ndarray
amrvac.read_uniform(path: str, *, resolution, bounds: tuple | None = None, field_indices: list[int] | None =
    None, ghost_width: int = 0, interpolation: str = 'zero', boundary_conditions=None) -> numpy.ndarray
amrvac.load_from_uniform(udata: numpy.ndarray, w_names: list[str], xmin: numpy.ndarray, xmax: numpy.ndarray,
    block_nx: numpy.ndarray, **kwargs)
amrvac.write_datfile(path: str, output_path: str, *, field_indices: list[int] | None = None, ghost_width:
    int = 0, overwrite: bool = False, boundary_conditions=None) -> dict
amrvac.write_datfile_from_uniform(file_path: str, udata: numpy.ndarray, w_names: list[str], xmin:
    numpy.ndarray, xmax: numpy.ndarray, block_nx: numpy.ndarray, overwrite: bool = False, **header_updates)
    -> dict
amrvac.load_uniform_data(file_path: str, field_indices: list[int] | None = None, return_geometry: bool = True)
amrvac.datfile_to_vtk(file_path: str, filename: str, field_indices: list[int] | None = None)
amrvac.openmp_build_info() -> dict[str, bool | int | str]
```

## 12. 数组场工具

```python
tools.potential_field_green(b3_bottom, xmin, xmax, nz, *, backend: 'str' = 'auto', balance_flux: 'bool' = True)
```

## 13. 解析构型

```python
configurations.bipolar_Avec(coordinates: numpy.ndarray, q_para: float, L_para: float, d_para: float)
configurations.bipolar_Bvec(coordinates: numpy.ndarray, q_para: float, L_para: float, d_para: float)
configurations.rbsl_Avec(coordinates: numpy.ndarray, x_axis: numpy.ndarray, a: float, F_flx: float, positive_helicity: bool)
configurations.TDm_slab(xmin, xmax, domain_nx, r0: float, a0: float, ispositive: bool, naxis: int, q0:
    float, L0: float, d0: float, knonb: float = 1.0)
configurations.dipolez_Avec(coordinates: numpy.ndarray, mz: float, posi: numpy.ndarray)
configurations.dipole_Bvec(coordinates: numpy.ndarray, m: numpy.ndarray, posi: numpy.ndarray)
configurations.monopole_Bvec(coordinates: numpy.ndarray, q: float, posi: numpy.ndarray)
configurations.fan_Avec(coordinates: numpy.ndarray, poses: numpy.ndarray, mz)
configurations.fan_Bvec(coordinates: numpy.ndarray, poses: numpy.ndarray, m: numpy.ndarray)
configurations.fan_slab(xmin: numpy.ndarray, xmax: numpy.ndarray, domain_nx: numpy.ndarray, poses:
    numpy.ndarray, m: numpy.ndarray)
configurations.curl_slab(vec: numpy.ndarray, dxyz)
```

## 高级接口的选择

`sm.select_source` 以借用适配器限制可用字段；`sm.cache_source` 显式缓存原始内部值；
`sm.plan_preparation` 复用网格几何计划。它们不是一次普通读取的前置步骤。
`sm.global_curl`、`global_curl_file` 提供分批或文件任务计算。

原生平面 LOS 的 `sm.integrate_los`/`integrate_los_views` 和
`integrate_thermal_los` 提供原始结果，以及各自支持的并行/参考实现控制。
只需可保存的射线结果时先使用上面的应用接口，避免混用结果类型。

`sm.bounded` 中的 `PreparedPool`、`trace_bounded`、`iter_traces_bounded`、
`sample_plane_bounded`、`iter_uniform_bounded`、`integrate_los_bounded` 是显式有限容量路径。
它们不使 QSL 或热 LOS 自动支持超内存输入。`iter_prepared` 的字段及视图在推进或关闭
迭代器后失效；必须在当前批次内消费，不能收集借用视图后统一计算。

需要检查任一高级接口的当前完整参数时，可直接查看其签名和说明：

```python
import inspect
from simesh import bounded
print(inspect.signature(bounded.PreparedPool))
help(bounded.integrate_los_bounded)
```

本页没有引入新函数、别名或统一执行器。接口简化建议与实测开销见
[应用接口审查](interface-review.md)。
