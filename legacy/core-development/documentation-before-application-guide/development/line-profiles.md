# 沿线物理量剖面

独立应用实现，基于共同基线 `4f6fc30`；分支 `codex/line-profiles`。
工作区为 `/Users/astery/.codex/worktrees/0d13/simesh`，只在其
`analysis-core/.venv` 中安装和运行。首次安装生成本工作区原生扩展，未改动核心
或 Cython 源码，未使用主工作区的扩展或父包源码。

## 交付接口

从 `simesh.line_profiles` 导入 `sample_line_profiles`、`LineProfiles` 和
`LineProfile`。主要调用：

```python
profiles = sample_line_profiles(
    quantities, lines, ("temperature", "density"),
    point_batch=4096, workers=4,
    length_units=sm.LengthUnits(1e6, "m"),
    memory_limit=2*1024**3,
)
branch = profiles.branch(seed_id, -1)
joined = profiles.line(seed_id)
```

完整英文调用说明和 Q/twist 筛选后追踪、读取剖面的示例见
[用户文档](../line-profiles.md)。

输入以现有 `LineSet` 为中心；通过原生采样内部入口和复用的线程池逐批读取
已完成 `Fields`，不重新追踪，不修改曲线点，不新增 AMR 插值逻辑。调用者可
直接传入含所需列的一个字段组；不依赖并行字段组合任务。

## 关键约定

- 所有原始点、种子编号、双支偏移和终止状态保持关联。逐点记录采样值、原始
  所属叶块、覆盖有效性、逐分量有限性及边界调整标记。覆盖有效与追踪完成独立。
- 每支弧长从首个种子点的零起点累计存储折线的欧氏长度；长度单位换算显式。
  不将其解释为精确 ODE 弧长或时间。合并视图按沿线正向排列，逆支弧长取负，
  正支取正；保留两份零弧长种子点、分支标记和原始点索引。
- 只有完全位于物理闭域内的精确上边界点，默认在临时采样副本中调整为最近
  域内浮点数；不夹回真正域外点，不补充缺失覆盖。原坐标与弧长不变。
  `boundary="native"` 保留原生半开域规则。
- 磁力线与采样场的来源分别保存，绝不要求值身份相同。`LineSet` 不含网格或
  时间描述符，调用者负责建立坐标、单位与时间的兼容性。字段借用期及有效晕区
  沿用原生检查；结果不保留 Source/Fields，可在来源关闭和借用过期后继续使用。
- 接收调用者提供的曲线时，每个非空支路首点必须等于其种子；方向和状态含义
  由调用者明确。有限但不可表示的距离累计或单位转换报错。
- 采样临时数组按点批次控制；预先估算并拒绝超出预算的保留输出。原生采样会
  读取输入组全部分量，再保留选中列，预算包括这些临时值。预算不是进程 RSS。

## 验证与整合边界

在本工作区 `analysis-core/` 执行：

```bash
.venv/bin/python -m pytest tests/test_line_profiles.py tests/test_applications.py tests/test_standard_diagnostics.py -q
```

结果：41 项通过，其中新增沿线测试 27 项。覆盖均匀与混合层级网格的解析温度/
密度和折线弧长、非连续和重合坐标的不同种子编号、单/双/空支路、部分追踪、
精确物理边界与单个浮点步长的真正域外点、缺失叶块、NaN/Inf 分量、不同批次/
工作者的逐值一致性、内存拒绝、借用过期与有效晕区检查、派生电流消费，以及
实际 Q/twist 筛选后独立双向追踪并读取温度/密度剖面的流程。

不修改共享 `__init__.py`、`applications.py`、`results_io.py`、`geometry.py` 或
全局 README/current 文档。公共导出、结果文件注册和字段组合交叉适配留给整合。
没有扩展 GUI 绘图、时间序列、大数据输入输出、真实强 QSL 验证或追踪算法。
