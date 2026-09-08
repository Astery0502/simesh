# 新一代 simesh：目标与资产继承

日期：2026-09-08。本文面向开发协作，记录本轮确认的建设范围和来源核对。

## 已确认目标

在仓库根目录的 `analysis-core/` 内建设独立的新一代 simesh，以
[intent](intent.md)、[共享场约定](prepared-fields.md)、
[科学结果](pipeline-results.md)及已有设计、估算和探索为依据重新组织实现。
用户已明确确认独立建设方向，并要求纳入各 worktree 的最新策略实现和
analysis-core 探索资产。此前“继续建设现有 simesh.analysis、不另建产品包”
只描述上一代实施范围，不再约束这次建设。

新一代的模块职责、数据模型、准备与消费边界、用户接口从这些需求推导。
已有代码提供可采用的算法与实现，不按旧目录逐文件搬运，也不因接口已经公开
就直接冻结为新接口。分别判断保留算法、保留执行组织和保留接口。

主工作流是选择字段与全域/区域、取得真实 AMR 支撑、一次准备、反复消费、
独立保留派生场并按需交付输出。效率评价包含读取、准备、计算和交付；
有界处理继续作为能力，缓存和存储政策不决定普通驻留用法。

`docs/analysis-core/` 继续保存定义、决策与证据，`scripts/analysis_core/`
中的实验与测量是资产来源。根目录 `analysis-core/` 是新的独立建设位置，
现已完成 N1–N4 独立可安装实现，参见[首阶段](evidence/next-generation-n1.md)、
[全域准备](evidence/next-generation-n2.md)与[科学交付证据](evidence/next-generation-n3.md)。
[N4 兼容交付](evidence/next-generation-n4.md)保留选定旧公开能力并建立显式数据边界。
现有 `src/simesh`、rewrite 和各 worktree 保留为资产及比较来源。

## 最新实现来源

2026-09-08 的 Git 与工作区核对结果：

| 来源 | 固定版本 | 继承关系与用途 |
| --- | --- | --- |
| `/Users/astery/.codex/worktrees/d538/simesh` | `b91bbc015d882ffd7dfcd7e10590cdce559f8532` | 最新完整实现来源；包含公开契约整理、结构整合、线程池统一和 B1–B5 后端改进 |
| `/Users/astery/.codex/worktrees/85cb/simesh` | `471f22068339abf77bc7d20e4a4075b2c08a2da4` | 上述版本的祖先；结构整合验收、构建与性能原始产物仍需从此处追溯 |
| `/Users/astery/.codex/worktrees/b403/simesh` | `5005c76dcdebe6155c776f46a691d05789809f25` | 效率收敛成果已进入最新来源；分叉后仅多一项 current.md 交接提交，无独有运行代码 |
| 主目录 | `d922e29` 加现有未提交改动 | 历史源码、原始数据、探索及测量产物；本轮新目标也记录在这里。不得以旧并行草稿覆盖后续已验收策略 |

三个来源 worktree 在核对时均无未提交改动。运行代码采用固定提交追溯，
忽略目录中的原始结果需独立登记来源，不能假定随 Git 提交一起迁移。
这里的登记不等于已经合并分支或复制原始产物。

## 要纳入的新策略与实现

| 资产 | 新一代建设时的处置 |
| --- | --- |
| 连续批量读取、普通字段选择和记录校验 | 采用最新读取资产；保留全域与区域读取的实际成本区别 |
| 独立字段、两层主场 halo、派生有效范围及发布语义 | 作为核心契约依据；具体类和模块名仍由新设计决定 |
| 真正的 NumPy base 所有权、发布前 coarse 释放、减少重复写入 | 采用后续修复，不能退回主目录较旧所有权实现 |
| 全域局部鬼点阶段无 GIL/线程池、区域 SAME/FINER 批量核 | 纳入最新执行资产；跨块阶段仍串行，粗层工作区/区域并行并未全部更新 |
| 定位提示、导数按块执行、采样与切片线程池 | 纳入已有优化及相应比较；四工作者是既有实现限制，不据此冻结新产品限制 |
| 追踪/twist、标量和热 LOS、编译 AMR 射线遍历 | 继承算法、结果语义与证据；保持线程池默认策略及显式 OpenMP 的既有区别 |
| 流式输出、可选轨迹、独立派生结果 | 纳入新接口设计；避免把输出分批和输入分页绑定 |
| 私有 v5、AMR 组件和编译 primitives | 最新版本已摆脱运行时 rewrite 包依赖；继续检查这些迁入组件的职责，不把迁入本身当作重新设计 |

最新实现的细节和证据可从来源提交的 `docs/analysis-core/current.md`、
`native-core-design.md` 及 `evidence/` 下的 `efficiency-consolidation.md`、
`backend-efficiency.md`、`threadpool-unification.md`、`structural-integration.md`、
`contract-convergence.md` 追溯。各轮数字只适用于其自身版本和负载。

## 探索资产也进入设计

纳入探索意味着保留问题、原型、证据、采用/回退结论和重新考虑条件，
不是把所有候选直接变成生产默认值。

- [E1 重分块](evidence/rebricking.md)：保留映射/消费原型、几何容量计算和
  稀疏放大证据。在冻结源叶块、计算块与存储槽身份前重新审视；直接砖边界
  准备和原生映射消费者尚未交付，不把历史暂缓等同于永久排除。
- [E5 几何计划](evidence/geometry-plans.md)：已有可选实现与重复准备收益。
  继承几何事实和数值状态分离；同一准备场反复消费无需重复执行计划。
- [E2–E4](algorithmic-directions.md)：保留重构、dual mesh/gridlets、RT 定位
  路线及进入条件。改变重构会改变科学含义，不作为同结果优化悄悄替换。
- [T01–T18](technique-candidates.md)、[数据组织](data-organization.md)、
  [生命周期](lifetime-sketches.md)及[准备与存储评估](preparation-reuse-review.md)：
  作为新边界设计输入；其中的历史实现缺口需要按最新源码重新判断。
- [运行时实验](runtime-execution.md)：保留准备块池、原始值缓存、多视角复用
  的条件性结果以及已回退方案。新设计有新证据时可重开，不重复无变化的实验。

## 仍需解决的继承边界

独立全域路径在 N2 已完成几何/最终值/工作区分离，来源中的旧 AMRMesh 留作
对照；区域粗层支撑仍有旧编排。全域 coordinate-phase 与区域 exact-phase
继续作为显式不同策略，不随覆盖选择自动切换。N4 已通过独立兼容层接入
旧 Dataset、普通值写回、2D 文件/数组、VTK 和工具；周期性仅保留文件元数据，
旧 Dataset 本来没有周期 ghost 邻接，现明确拒绝这种计算。CT 普通 writer
也不再生成声明存在但实际缺失 CT 载荷的文件。

Q、精确足点、新分析的 2D/周期/CT/GPU、有限内存非线性热辐射及真实大输入
验收仍有缺口。历史制造温度图像不是真实快照热辐射。记录这些缺口不等于
把它们全部加入第一阶段实现队列，也不缩减原有产品意图。

## 首版交付与后续

首版[模块、接口与实现取舍](next-generation-design.md)已建立上述设计映射，
确定 Source/准备分离、存储偏移/有效范围分离、显式数值 scheme、独立构建及
N1–N4 实施检查点。N1 的来源/区域准备/直接消费和 N2 的全域准备/所有权拆分
以及 N3 的科学消费者、计划、有界与大输出均已实现并验收。N4 已接入旧公开
Dataset、写回、建文件、导出和独立数组工具，并完成独立安装及显式跨边界交付。
各项数字和数值修正见 [N1](evidence/next-generation-n1.md)、
[N2](evidence/next-generation-n2.md)、[N3](evidence/next-generation-n3.md) 和
[N4](evidence/next-generation-n4.md)。N1–N4 首版已闭环；后续科学/规模目标
仍按各自范围推进，不把交付等同于所有历史内部接口或长期能力均已替换。
