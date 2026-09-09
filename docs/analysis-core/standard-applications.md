# 标准诊断与应用产品

日期：2026-09-09。面向开发协作。

在点/射线/结果关联层之上补充了常用入口，没有改变“QSL/twist 不保存路径、
筛选后单独追踪、应用追踪默认双向”的决定。

## 已实现

- 标准场：`magnitude`、`gradient`、`divergence`、`dot`。
- 物理场：`MagneticUnits` 明确指定磁场与长度的 SI 换算，
  `current_density` 返回 A/m²，`magnetic_pressure` 返回 Pa，
  `magnetic_energy_density` 返回 J/m³。磁导率为显式配置项，默认采用常规真空近似。
- 常见产品：`applications.uniform_grid` 收集均匀场与几何；`field_map` 生成面上场图；
  `surface_diagnostics` 和 `bottom_diagnostics` 提供任意面与底面的磁诊断。
- `line_diagnostics` / `iter_line_diagnostics` 和应用层支持只算 Q、只算 twist 或联合请求。
  只算 twist 时，仅积分中心线与 curl；不追踪邻线、不构造单位场梯度、不计算 Q。
  未计算的 Q 数组为 None，有效性与筛选按实际请求的量处理。

物理边界处的导数仍遵循准备时的鬼点方案。标准物理量接口不会把连续外推的
边界导数重新解释成其他物理边界条件，也不会根据字段名称猜测单位。

## 检查与示例

完整测试集为 63 项通过、2 项因普通构建未启用 OpenMP 跳过。
新增测试覆盖已知解析梯度/散度、SI 电流换算、磁压与能量单位、均匀输出几何和预算，
以及单独 twist 与联合请求的端点和 twist 一致性。
测试还显式禁用了 Q 的相关函数，确认 twist-only 没有依赖这些计算。

`analysis-core/examples/standard_applications.py` 已运行完整合成案例：

- 9 个混合层级 AMR 叶块，生成磁场和密度并复用准备结果。
- 电流图、磁压均匀场、768 点的底面 Q/twist 图。
- 阈值筛选出 96 个种子，另行双向追踪，共保存 49,524 个实际路径点。
- 标量 LOS 和 AIA171 热 LOS 各输出 2,304 个有效结果，包含没有域交叠的空视线。
- 导出 NPZ、含单位/几何/状态的 JSON，并生成两张 PNG。

数值计算使用独立环境。绘图依赖下载遇到网络 SSL 错误后，使用本机已有 Matplotlib
仅渲染导出的数据；示例的计算部分不依赖 Matplotlib，绘图依赖作为可选 `plot` 配置。
未加入指向父环境的运行依赖。

该案例的物理尺度与等温假设仅为演示，不是实际快照标定，也不是强 QSL 或大输入验收。
外部调用说明见[标准应用](../../analysis-core/docs/standard-applications.md)。
