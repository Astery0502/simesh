# simesh 主要应用场景介绍

本文帮助使用者按科学问题选择分析方法，说明当前支持的应用、所需输入、主要接口和输出限制。参数定义以生成的接口参考为准。

参数细节见[应用接口](user/api.md)。下文统一采用以下导入；接口表中的调用只突出关键参数，不是可直接独立执行的完整脚本。

```python
import simesh as sm
from simesh import applications as app, tools
from simesh.tools import configurations
```

## 1. 应用范围与任务地图

simesh 主要用于 AMR 模拟数据的后处理：在原生网格上采样、求导、追踪向量场、计算磁连接性、恢复物理量和积分，也提供数组形式的势场与解析磁场工具。适合围绕一个快照组织分析，再由调用脚本将相同流程用于多个快照。

| 编号 | 应用场景 | 要回答的问题 | 主要交付物 |
| --- | --- | --- | --- |
| A01 | 点采样、切片与均匀重采样 | 某位置、截面或区域内的场如何分布？ | 数值图、采样点表、均匀三维数组 |
| A02 | 派生场与磁场局部诊断 | 电流、磁压、磁能以及场的空间变化在哪里？ | 可继续采样、积分的派生字段 |
| A03 | 磁力线与速度流线 | 指定位置的向量场沿哪条空间曲线延伸？ | 带种子编号和终止状态的曲线 |
| A04 | Q、Q⊥、twist 与磁连接性 | 磁力线映射在哪里剧烈变化，哪些种子值得进一步分析？ | 诊断图、边界脚点、筛选后的种子 |
| A05 | 理想 MHD 状态恢复 | 守恒量对应怎样的速度、热压、温度和特征速度？ | 采用明确物理模型和单位的状态字段 |
| A06 | 原生 AMR 积分、统计与面通量 | 区域总量、平均量、分布和穿面通量是多少？ | 带单位和覆盖信息的定量结果 |
| A07 | 沿线物理量剖面 | 密度、温度、电流沿已选曲线如何变化？ | 弧长—物理量剖面 |
| A08 | 标量视线积分 | 给定方向上的柱密度或其他标量积分是什么？ | 带射线状态的投影图 |
| A09 | EUV 与射电辐射合成 | 指定热力学状态和吸收模型会形成怎样的多波段图像？ | 带模型、单位、光深和有效性的辐射图 |
| A10 | 势场外推与解析磁场 | 如何从底面磁场构造参考场，或生成可控的分析样本？ | 均匀磁场数组、矢势或解析构型 |
| A11 | 数据读写与结果交付 | 如何准备输入、保存结果、交给其他程序继续使用？ | `.dat`、应用结果文件、分片目录等 |
| A12 | 根块组合区域分析 | 如何提取多个 level 1 根块并独立研究其中的精细结构？ | 保留细化层级的子区域文件、局部分析结果 |

按分析依赖选择流程：A01/A02 提供场分布和局部诊断，A03/A04 提供曲线和连接性；A05 的恢复量可用于 A06、A07、A09，A03 的曲线是 A07 的输入。A10 提供参考磁场，A11 负责各条流程的数据读写；A12 将根块组合成子区域，可接入前述分析流程。

### 共同输入约定

原生科学分析支持满足平衡条件的 Cartesian 三维 AMR 网格，以及 AMRVAC v5 普通单元中心字段；等价数组也可建立输入。逐轴周期配置仅用于 `exact-phase` 的 halo 准备，由对侧真实叶块提供鬼点；非周期侧默认连续外推，也可通过 Source 的 `boundary` 按字段和面显式设置对称或反对称镜像。坐标、采样、区域与积分保持原有有限域语义，不支持周期轨迹和磁连通性分析。`coordinate-phase`、AMRVAC 与均匀网格导出、AMR 切片结果保存均拒绝周期网格。球坐标、CT 面变量和 GPU 分析仍不在支持范围内。

| 对象或入口 | 应用中的作用 |
| --- | --- |
| `sm.open_amrvac(path, fields=...)` | 从文件建立 `Source`，选择需要读取的字段 |
| `sm.mesh_from_forest(...)`、`sm.source_from_arrays(...)` | 用网格描述和叶块内部数组建立合成数据输入 |
| `sm.read_fields(source, ...)` | 得到内部单元值，适合体积分、统计和不依赖邻点的恢复公式 |
| `sm.prepare(source, ..., scheme=...)` | 得到带有效邻域支撑的 `Fields`，供插值、导数、路径和视线计算使用 |
| `sm.PointSet`、`sm.Plane`、`sm.RaySet` | 定义可复用的采样位置、平面和射线几何 |

`coordinate-phase` 用于全域准备；盒选区准备使用 `exact-phase`。两者有不同数值定义，比较分析结果时需固定方案。`prepare` 提供两层有效 halo，插值至少需要一层，一阶差分消耗一层。区域边缘不是物理边界；轨迹可能越出准备区域，消费者不会自动读取缺失数据。

数组输入布局为 `(leaf, component, x, y, z)`，`Fields.values` 布局为 `(slot, x, y, z, component)`。读取内部值用 `interior()`，避免把 halo 计入物理统计。字段单位标签不执行数值换算；磁场、密度、长度、能量等尺度必须与实际数据一致。

## 2. A01：点采样、切片与均匀重采样

**简介。** 将原生 AMR 场变成指定点、平面或规则体网格上的数值，适合场分布展示、截面对比和为其他数组程序准备输入。可以直接采样原始字段，也可以采样 A02、A05 得到的连续派生量。

**输入与接口。** 输入是已有有效插值支撑的 `Fields`，以及坐标、平面或输出分辨率。

| 入口 | 功能与输出 |
| --- | --- |
| `sm.slice_axis(fields, axis, coordinate, components=...)` | 从内部单元直接提取原生 AMR 截面，返回可保存的 `AMRSliceResult`，无需插值 halo |
| `app.sample(fields, points, components=...)` | 对 `PointSet` 采样，返回 `SampledPoints`，保留点编号、值和覆盖 |
| `app.field_map(fields, surface, components=...)` | 接受 `Plane` 或 `PointSet`，返回相同类型；有图像布局时可访问 `image` |
| `app.uniform_grid(fields, resolution, components=..., bounds=...)` | 返回 `UniformResult`，数值布局为 `(nx, ny, nz, component)` |
| `sm.sample(fields, positions, components=...)` | 仅需数组时，返回 `(values, owners, valid)` |
| `sm.iter_uniform(fields, resolution, ...)` | 按 z 切片交付 `(z_index, SliceResult)`，供大体数据逐片写出 |

平面的 `u`、`v` 向量表示整幅图的跨度，采样位于像素中心。`valid` 表示覆盖，应用结果的 `usable` 还要求所选分量数值有限。提高输出图像分辨率只增加采样点，不增加原始模拟的物理分辨率。

参考：[应用接口](user/api.md)。

## 3. A02：派生场与磁场局部诊断

**简介。** 从原始变量计算自定义逐点公式、梯度、散度、旋度，以及带物理归一化的电流、磁压和磁能密度。可用于定位电流集中区域、展示能量分布或检查磁场散度。

**输入与接口。** 逐点代数运算只需要共同覆盖的字段；求导需要有效 halo。多个输入组必须共享同一个 `Mesh` 对象和叶块覆盖。

| 入口 | 功能与输出 |
| --- | --- |
| `sm.derive(...)`、`sm.derive_many(...)` | 单个或多个逐点公式，返回拥有独立存储的 `Fields` |
| `sm.magnitude(...)`、`sm.dot(...)` | 模长与点积 |
| `sm.gradient(...)`、`sm.divergence(...)`、`sm.curl(...)` | 标量梯度、向量散度、向量旋度 |
| `sm.current_density(fields, units=..., components=...)` | 计算 `J = curl(B)/mu`，返回 A/m² |
| `sm.magnetic_pressure(...)`、`sm.magnetic_energy_density(...)` | 计算 `B²/(2mu)`，分别标记 Pa 和 J/m³ |

物理磁场诊断使用 `sm.MagneticUnits(field_tesla=..., length_m=...)`。通用导数使用原坐标长度，不自动完成 SI 换算。`derive` 回调只适合逐点公式，空间移位、求导和归约应使用相应接口。若要求导后再采样，输入通常需两层有效 halo。

参考：[采样与字段接口](user/api.md#sampling-and-fields)。

## 4. A03：磁力线与速度流线

**简介。** 从一组种子出发追踪三分量向量场的空间曲线。磁场用于磁力线展示；由 A05 恢复的速度可用于瞬时流线展示。速度流线描述当前快照的方向结构，不包含随时间演化的粒子轨道。

**输入与接口。** 输入必须是恰好三个分量的已准备向量场，以及 `PointSet`；`step` 是原坐标中的长度步长。

| 入口 | 功能与输出 |
| --- | --- |
| `app.trace(fields, points, step=..., direction=..., max_steps=...)` | 返回紧凑的 `LineSet`，默认保存沿场和逆场两支 |
| `app.iter_lines(fields, points, seed_batch=..., ...)` | 按种子批次返回 `LineSet` |
| `lines.branch(seed_id, -1)`、`lines.branch(seed_id, +1)` | 按稳定种子编号取得单支坐标 |
| `lines.line(seed_id)` | 拼接该种子的两支曲线，便于显示 |
| `sm.trace(...)`、`sm.iter_traces(...)` | 原始追踪接口；默认单支且不保留轨迹，保存轨迹需 `trajectories=True` |

应用入口的 `direction` 可选 `both`、`along`、`against`、`inward`；`inward` 要求种子位于单一物理边界面且场不与其相切。输出保留种子、路径坐标和两支 `termination`。步数或长度用尽、零场、缺覆盖等终止与正常出域必须分别统计。保存的是最后接受的路径前缀，末点不保证精确落在边界。

参考：[应用接口](user/api.md)、[恢复状态与速度流线示例](../examples/recovered_state_analysis.py)。

## 5. A04：Q、Q⊥、twist 与磁连接性

**简介。** Q 描述磁力线脚点映射的拉伸程度，可用于寻找准分离层候选结构；Q⊥提供垂直于场方向的映射诊断；twist 提供沿磁力线的扭转诊断。将诊断图与有效性结合，可挑选种子后继续追踪和做沿线分析。高 Q 或高 twist 的阈值应由具体科学问题确定，不能直接当作重联发生或结构不稳定的结论。

**输入与接口。** 输入是三分量磁场，准备范围需覆盖积分路径及所选方法需要的邻域。各应用入口统一通过 `quantities="q"`、`"twist"` 或 `("q", "twist")` 选择计算内容。默认计算两者；兼容的 `twist=False` 仅在未指定 `quantities` 时可用，不能同时传入两个选择参数。

| 入口 | 功能与输出 |
| --- | --- |
| `app.bottom_diagnostics(magnetic, shape, quantities=..., ...)` | 在物理 z 下边界布点，返回 `ConnectivityMap` |
| `app.surface_diagnostics(magnetic, surface, ...)` | 在指定平面或点集布置诊断种子 |
| `app.connectivity(magnetic, points, ...)` | 对已有编号种子计算诊断 |
| `diagnostic.image("log10_q")`、`diagnostic.image("twist")` | 按种子布局取得诊断图 |
| `diagnostic.threshold(q_min=..., abs_twist_min=..., mode=...)` | 按有效性和包含等号的阈值筛选，返回 `PointSet` |
| `sm.qsl(...)`、`sm.line_diagnostics(...)` | 返回原始 `QSLResult`，供需要数组和完整诊断字段的程序使用 |

结果含请求的诊断量、脚点、长度、边界编号与终止状态。画 Q 图用 `q_valid`，画完整线 twist 图用 `twist_valid`；没有请求的量不应继续读取。诊断阶段不保存全路径，选中种子后用 A03 生成曲线。采样平面定义的是种子位置，不自动成为目标脚点边界。

QSL 使用邻种子脚点差分，不提供算法选择参数。`normalization="mapping"` 与 `"flux"` 是不同归一化选择。`delta` 控制邻种子扰动距离，`step_fraction` 控制局部积分步长，两者应分别检查收敛。twist 若显式传入 `curl_field`，必须使用匹配的原始 `sm.curl(magnetic)`，不能换成物理电流。

参考：[应用接口](user/api.md)。

## 6. A05：理想 MHD 状态恢复

**简介。** 将密度、动量、能量和磁场转换为便于物理分析的状态量，为温度图、速度流线、质量加权统计和热辐射合成提供输入。

**输入与接口。** 主要入口是 `sm.mhd_fields(conserved, model=..., magnetic=..., outputs=...)`。用 `sm.IdealMHD` 指明 `gamma`、`energy_kind`、组成和 `sm.MHDUnits`；字段选择参数允许适配实际名称。守恒量和磁场可以分组提供，但网格对象和覆盖须一致。

| 可选输出 | 用途 |
| --- | --- |
| `density`、`velocity`、`speed` | 质量统计、速度流线和速度图 |
| `internal_energy`、`pressure`、`temperature` | 内能、热压、热结构和辐射输入 |
| `beta`、`sound_speed`、`alfven_speed` | 热压与磁压比较、特征速度分析 |
| `sonic_mach`、`alfven_mach`、`status` | 马赫数与状态分类 |

返回 `Fields`，密度为 g/cm³，速度为 cm/s，内能密度为 erg/cm³，热压为 dyn/cm²，温度为 K。模型适用于经典动量 `m=rho*v`、恒定 gamma、完全电离 H/He，以及明确的总能量或内能定义。背景磁场分裂、额外能量库或其他状态方程不能仅凭字段名自动兼容。

默认 `invalid="raise"` 拒绝非法物理状态；选择 `"nan"` 时应请求并查看 `status`。零磁场可以是有效状态，但 beta 等比值可能未定义。状态列是分类数据，不能参与插值。仅做单元统计时可从 `read_fields` 恢复；后续需要采样或流线时应从已准备的输入恢复连续量。

太阳日冕归一化可直接调用；这里的数密度单位指氢核数密度，需与模拟设置一致：

```python
composition = sm.CoronalComposition(helium_abundance=0.1)
units = sm.MHDUnits.solar(composition=composition)
model = sm.IdealMHD(gamma=5/3, energy_kind="total", composition=composition, units=units)
state = sm.mhd_fields(conserved, model=model)
```

归一化约定与 AMRVAC 的 CGS、完全电离 H/He、`eq_state_units=True` 相容。参数与默认值见 [MHDUnits](user/api.md#simesh.MHDUnits)。恢复密度传给 `thermal_fields` 时使用 `density_unit_g_cm3=1.0`；积分长度使用 `sm.LengthUnits(units.length_cm, "cm")`。独立的磁场诊断仍输出 SI，可通过 `units.magnetic_si` 显式提供换算。

参考：[应用接口](user/api.md)。

## 7. A06：原生 AMR 积分、统计与面通量

**简介。** 直接在 AMR 内部单元上计算质量、磁能、平均温度、极值、分布和矩形面的磁通等量。真实单元体积参与权重，适合包含多级细化的定量比较。

**输入与接口。** 输入是已换算到所需字段单位的内部值，可直接使用 `read_fields`；`sm.LengthUnits(scale, unit)` 负责坐标长度换算。

| 入口 | 功能与输出 |
| --- | --- |
| `sm.volume_integral(fields, component, region=..., units=...)` | 盒选区体积分，返回带单位及覆盖的 `ScalarResult` |
| `sm.weighted_mean(fields, component, weights=..., weight_component=..., ...)` | 默认体积平均；以密度为权重可得质量加权平均 |
| `sm.extrema(fields, component, ...)` | 极值、位置及所属叶块/单元记录 |
| `sm.histogram(fields, edges, component, ...)` | 显式分箱的加权分布，另保留下溢和上溢贡献 |
| `sm.surface_flux(fields, surface, component, units=...)` | 指定标量法向分量与有向矩形面积的积分 |
| `sm.AxisAlignedSurface(axis, coordinate, bounds, normal=..., side=...)` | 指定轴对齐矩形面、法向符号及内部面取值侧 |

区域裁剪使用单元交叠体积或面积，场采用给定内部值的分片常数表示。面通量需由调用者选对法向分量；它是单元中心场的单侧估计，不提供 CT 面通量或粗细网格通量连续性保证。

默认拒绝缺覆盖及非有限值；使用 `missing="omit"` 或 `nonfinite="omit"` 后，数值仅描述有效覆盖，需一起保存 `coverage`。归约结果可直接用 `save_result` 保存，再用 `load_result(...).result` 恢复；值、单位、覆盖、权重、极值位置、面方向和直方图尾部均随对应结果保留。

参考：[应用接口](user/api.md)。

## 8. A07：沿线物理量剖面

**简介。** 在已生成或恢复的曲线上采样其他物理量，不必重新追踪。适合比较磁力线上的温度、密度和电流，也可对速度流线进行相同分析。

**输入与接口。** 使用 `sm.sample_line_profiles(fields, lines, components=..., length_units=..., point_batch=...)`，返回 `LineProfiles`；大规模曲线可用 `sm.iter_line_profiles`。输入字段需要有效插值支撑，曲线为 `LineSet`，两者的快照时间、坐标与物理尺度需由调用者确认兼容。

结果保留种子编号、分支、坐标、弧长、数值及有效性。用 `profiles.branch(seed_id, -1/+1)` 取得单支剖面。每支弧长从种子计零，`length_units` 只转换弧长，存储坐标仍保持原尺度。`usable` 按采样分量记录，不能替代原曲线的终止状态检查。

参考：[应用接口](user/api.md)。

## 9. A08：标量视线积分

**简介。** 沿指定观察方向计算标量积分，例如质量柱密度、标量发射率投影或不同视角的结构对比。它计算选定场的线积分，所需物理解释由字段定义决定。

**输入与接口。** `sm.RaySet.from_plane(plane, direction, near=..., far=...)` 建立射线，`app.los(fields, rays, component=0, quadrature=..., ...)` 返回 `RayResult`。`sm.orthographic_plane(lower, upper, direction, shape)` 可自动构造覆盖盒投影的相机平面。原始图像接口为 `sm.integrate_los` 和 `sm.integrate_los_views`。

`component` 支持字段名或局部分量索引，建议用名称避免字段重排造成误选。标量结果单位是字段单位乘原坐标长度，实际柱密度还需显式乘相应长度换算系数。`near/far` 控制沿射线的深度截断。输出包含积分值、入域/出域深度、状态和采样次数，`image` 提供图像布局。

默认 `quadrature="gauss2"`，也支持 `"midpoint"`。空射线可以是合法零贡献；应通过 `status` 区分空射线、正常积分、缺覆盖和采样上限。`complete` 表示全部射线具有有效状态。

参考：[应用接口](user/api.md)。

## 10. A09：EUV 与射电辐射合成

**简介。** 将密度和温度转换为热力学场，可合成 AIA、IRIS、EIS 的 12 个 EUV 通道，以及射电自由–自由辐射。光学厚路径显式加入吸收，返回亮度、光深和无吸收亮度。

**输入与接口。** 密度必须明确数值到 g/cm³ 的系数；温度以 K 提供，可以是显式等温常量，也可以来自 A05 的恢复场。`density_component` 与 `temperature_component` 均支持字段名或局部分量索引。

| 入口 | 功能与输出 |
| --- | --- |
| `sm.EUV(wavelength=...)`、`sm.RadioFreeFree(...)` | 选择波段及完全电离发射模型；`sm.AIA171(...)` 保留历史约定 |
| `sm.thermal_fields(density, temperature, density_unit_g_cm3=..., temperature_label=..., ...)` | 生成数密度和温度字段，记录温度解释 |
| `sm.emissivity_fields(thermodynamics, model=...)` | 生成节点发射率，供局部分布检查等用途 |
| `app.thermal_los(thermodynamics, rays, length_unit_cm=..., order=..., subdivisions=...)` | 返回 EUV 光学薄 `RayResult`，亮度单位为 DN s^-1 pixel^-1 |
| `sm.radiation_fields(thermodynamics, model=..., absorption=...)` | 生成发射率和吸收系数；EUV 可选 `sm.HHeAbsorption()`，射电使用自身吸收 |
| `sm.radiative_los(coefficients, rays, length_unit_cm=...)` | 沿观察者到远端的方向求解传输；射电亮温单位为 K |

默认 `order="thermodynamics-first"` 先插值密度和温度，再计算非线性响应；`"emissivity-first"` 先计算节点发射率再插值，是不同重构。`subdivisions` 控制非线性求积细分，不改变所选重构。输入能量列不会被此接口自动解释为温度。

新 EUV 默认发射量为 `nₑ nH R(T)`；原 `AIA171` 默认保持 `nₑ² R(T)`。吸收模型只用于辐射后处理，不改变输入热力学状态。厚辐射先在节点计算系数再插值，应增加细分检查收敛。1 MK 等温图中的 EUV 吸收很弱，不能替代冷吸收层验证。

参考：[热力学与视线接口](user/api.md)、[多波段示例](../examples/radiation_bands.py)。

## 11. A10：势场外推与解析磁场

**简介。** 从二维底面法向磁场构造三维势场参考，或生成双极场、偶极场、磁通绳等可控构型。适合构造分析输入和参考场，也可用于比较给定磁场与参考构型的形态。

**输入与接口。** 此组接口直接处理 NumPy 数组，不要求 `Source` 或 `Fields`。

| 入口 | 功能与输出 |
| --- | --- |
| `tools.potential_field_green(b3_bottom, xmin, xmax, nz, backend=..., balance_flux=...)` | 底面场的 Green 核外推，返回 `(3, nx, ny, nz)` 磁场和几何信息 |
| `configurations.bipolar_Bvec(...)`、`dipole_Bvec(...)`、`monopole_Bvec(...)` | 双极、偶极、单极磁场 |
| `configurations.rbsl_Avec(...)`、`TDm_slab(...)` | RBSL 矢势和 TDm 构型 |
| `configurations.fan_Bvec(...)`、`fan_slab(...)` | fan 构型及均匀体数据 |
| `configurations.curl_slab(...)` | 均匀数组上的旋度辅助运算 |

外推的 `direct` 与 `fft` 对应同一中点源卷积的不同实现，后者需要 SciPy。默认 `balance_flux=True` 会移除底面均值，改变外推使用的边界场；几何信息记录被移除的均值。势场外推不包含电流驱动的非势场重建。

解析构型的坐标通常为分量在前的数组，不能直接当作 `(n, 3)` 点表。要接入 A01—A09，需核对坐标、单元中心位置、分量轴、网格描述和单位后建立 Source。

参考：[数组工具接口](user/api.md#array-tools)。

## 12. A11：数据读写与结果交付

**简介。** 将分析输入与计算结果变成可重复使用的数据产品，便于后续绘图、跨进程分析和大批次交付。不同输出格式保留的信息不同，保存时需明确文件的读取入口。

| 任务与入口 | 功能和范围 |
| --- | --- |
| `sm.open_amrvac(...)`、`sm.read_fields(...)` | 读取笛卡尔三维 AMRVAC v5 普通场，保留仅用于 halo 准备的逐轴周期标记，数值采用分量在后布局 |
| `sm.source_from_arrays(...)` | 从明确的网格与块数组建立 Source |
| `sm.export_uniform(...)` | 将普通场导出为均匀体数据，支持按批读取 |
| `sm.write_uniform_vtk(path, grid)` | 将已有均匀体结果保存为二进制 VTK，保留网格边界、标量分量与覆盖标记 |
| `sm.export_uniform_vtk(source, path, resolution, ...)` | 直接从原始文件或 Source 重采样并生成均匀网格 VTK，复用均匀场接口的采样控制 |
| `sm.save_result(path, result, metadata=...)`、`sm.load_result(path)` | 保存和恢复支持的几何、原生截面、积分统计、采样、连接性、射线、曲线、剖面及均匀结果对象 |
| `sm.save_result_shards(...)`、`sm.open_result_shards(...)` | 逐批保存并按分片加载曲线/剖面等支持的批次结果 |
| `sm.write_amrvac(path, fields, metadata=..., root_bounds=...)` | 将完整原网格或根块对齐区域的字段导出为普通 `.dat` |
| `sm.select_roots(...)`、`sm.crop_amrvac(...)` | 选择完整根块子树，或直接从文件裁剪为独立快照，见 A12 |

当前包仅提供 Source/Fields 数据流程。VTK 导出仅支持均匀体，不保存 AMR 层级或曲线；每个分量保存为单元标量，`simesh_valid` 单独记录覆盖有效性，单位和来源信息需另行保存。历史 Dataset 和二维数据流程不在当前发布范围内；旧代码见 `legacy/previous/`。

`load_result(...).result` 恢复保存的应用对象；自定义 `numpy.savez_compressed` 文件需用 `numpy.load`。原始 `Fields`、`TraceResult` 和原始 LOS 结果不能直接传给 `save_result`；曲线和投影应使用应用入口返回的结果。原生截面保存原始网格拓扑以重建块编号和单元边界，整体保存和加载不保证有限内存。额外数值控制、模型和来源应放入 `metadata` 等记录中；加载器不会核验结果与原始快照的物理对应关系。

大曲线使用 `app.iter_lines`，大均匀体使用 `sm.iter_uniform`。分批输出不意味着所有算法都支持有限容量输入，尤其 QSL 与热 LOS 仍要求输入字段驻留。分片清单完成也不代表所有积分成功，分片不是积分状态断点。

参考：[数据产品与保存协议](user/api.md#outputs)、[高级执行](dev/api.md#preparation-and-bounded-execution)。

## 13. A12：根块组合区域分析

**简介。** 以 level 1 根块为单位，将相邻根块组成矩形区域，提取完整细化子树。区域内部的细化层级、单元间距和字段数值保留，可独立分析局部结构，也可由脚本组织多个区域进行对比。

**输入与接口。** `root_bounds` 是零基整数上下界，上界不包含。下面选取 `3 × 2 × 2` 个根块；输入网格必须包含这一范围。

```python
box = ((1, 1, 0), (4, 3, 2))
sm.crop_amrvac("snapshot.dat", "crop.dat", root_bounds=box, fields=("rho", "b3"))
with sm.open_amrvac("crop.dat") as source:
    local = sm.read_fields(source)
section = sm.slice_axis(local, "z", float(local.mesh.lower[2]))
sm.save_result("crop-section.result.npz", section)
```

若需要保留原域的邻域支撑，先用 `select_roots` 建立选区，再显式准备：

```python
with sm.open_amrvac("snapshot.dat") as source:
    selection = sm.select_roots(source.mesh, box)
    ready = sm.prepare(source, ("rho", "b3"), region=selection, scheme="exact-phase")
local_grid = app.uniform_grid(ready, (48, 32, 32), bounds=selection.requested_bounds)
```

这两条路径的边界含义不同：内存选区仍属于原始网格，准备时可读取原域邻块；独立裁剪文件的边缘则成为新计算域边界，不保留原域外侧 halo。求导或追踪时应选足邻域与路径范围，不能将裁剪边缘解释为原模拟的物理边界。区域积分默认使用选区；均匀输出需显式指定区域范围，原生切片则保留所属网格的完整截面几何和覆盖标记。

`write_amrvac(..., root_bounds=box)` 还可导出已读取或派生的字段。裁剪不会重采样成 level 1 均匀网格；文件直接裁剪按批处理选中字段，输入网格和索引仍驻留内存。

参考：[根块区域分析](user/index.md#root-block-regional-analysis-and-independent-output)、[可执行裁剪示例](../examples/root_crop.py)。

## 14. 可直接运行的示例

这些示例自行生成输入，可先用来熟悉输出格式，再替换成自己的快照和单位配置。

| 现有入口 | 可复用内容 | 结果读取 |
| --- | --- | --- |
| [快速开始](../examples/user_quickstart.py) | 教学快照、磁场图、质量积分及保存恢复 | 应用结果用 `sm.load_result` |
| [标准应用示例](../examples/standard_applications.py) | 合成磁场、电流图、Q/twist、筛选曲线、标量/热 LOS；可选绘图 | `products.npz` 用 `numpy.load`，另有 `summary.json` |
| [根块裁剪示例](../examples/root_crop.py) | 根块选区准备、独立文件裁剪及截面和积分保存 | `.result.npz` 用 `sm.load_result`；裁剪文件用 `sm.open_amrvac` |
| [恢复状态分析示例](../examples/recovered_state_analysis.py) | MHD 恢复、质量/温度/磁通、热图、速度流线和沿线量检查 | 状态统计、热图、流线和剖面均用 `sm.load_result` 恢复 |

在仓库根目录运行，输出目录使用本次新目录：

```bash
.venv/bin/python examples/user_quickstart.py --output /tmp/simesh-overview-quickstart
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-overview-standard
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-overview-recovered
.venv/bin/python examples/root_crop.py --output /tmp/simesh-overview-crop
```

恢复状态示例提供的已知量为质量 `2.4e12 g`、质量加权温度 `840000 K`、底面向外磁通 `-2e18 G*cm²`。这些值来自该示例的解析输入，不能作为其他数据的通用验收标准。
