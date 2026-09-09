# 当前方向：新版提升为根目录核心

更新：2026-09-09。面向开发协作。

用户已确认固定现有 N4 核心设计。后续重点是应用层如何选择、组合、呈现和
交付已有数据与计算结果。随后按用户需求增加了原生派生量组合与磁连接性消费者。

- 选定实现：根目录 `src/simesh/`，由原 `analysis-core/` 整包提升；
  在 N4 基线 `a817221` 上增加的应用接口与数值边界保留。
- 上一代主包与 `rewrite/` 位于 `legacy/previous/`，更早的 Python 实现位于
  `legacy/python-first/`；两者不进入当前构建、导入和默认测试。
- 已移除当前包的 `utils/lib/amr` 层级：仍供 Dataset 使用的 Cython 代码收拢到
  `amrvac/_mesh/`，数组构型移到 `tools/configurations.py`。
  旧导入路径不保留转发；参见根目录[迁移说明](../../../../MIGRATION.md)。
- 目录迁移、根目录安装与隔离 wheel 验证已完成，见[迁移记录](root-promotion.md)。
- 面向用户的[物理功能与输出清单](../capabilities-and-outputs.md)
  汇总现有能力、兼容边界、文件保存/读取方式和两个代表性示例的实际产物。
  交付范围以这份清单和专题文档为准；早期应用复查中的已完成建议不再作为待办。
- 三个独立 feature 已整合：原生积分与统计、显式 MHD 热力学恢复、应用结果保存恢复。
  公共导出和联合流程已检查，见[整合记录](feature-integration.md)及
  [联合调用示例](../quantitative-workflow.md)。
- 原生 `derive` 支持逐点公式组合，`derivative` 支持字段名与方向名。
- 字段组合与沿线诊断已完成整合及 simplify：`select_fields`、`merge_fields`、
  `derive_many` 和 `sample_line_profiles` 已公开导出。
  见[本轮整合记录](composition-profile-integration.md)。
- 标准诊断提供模长、梯度、散度、点积，以及显式 SI 归一化的电流、磁压和磁能密度。
  面上诊断支持 Q-only、twist-only 和联合请求；twist-only 跳过 Q 的计算。
  统一示例覆盖均匀场、电流图、诊断筛选后双向追踪，以及标量/热 LOS。
  见[标准应用](../standard-applications.md)。
- `PointSet`、`RaySet` 和 `simesh.applications` 提供点/射线编号、几何与结果关联。
  按用户最新决定，QSL/twist 不保存积分路径；阈值筛选后再单独追踪。
  用法见[应用接口](../applications.md)。
- `qsl` / `iter_qsl` 支持 AMR 上的 Q、Q⊥、局部映射、边界脚点与完整线 twist；
  方法、状态与误差边界见[实现检查](connectivity.md)和[调用说明](../connectivity.md)。
- 设计边界：[固定核心设计](next-generation-design.md)。
- 应用开发：[简短指引](application-development.md)及新包
  [README](../../../../README.md)、[MIGRATION](../../../../MIGRATION.md)。
- 两个采样复用候选均未采用，相关 worktree 已移除；不自动恢复这些探索。
- 旧源码已归档，其他历史 worktree 与既有验收产物保留，各自的实现范围不混用。

一般应用改动按直接源码检查、一个有代表性的消费流程和必要用例推进。
不默认要求阶段计划、独立评审、完整新旧性能比较或多任务并行。
涉及物理量、单位、覆盖或输出语义时，明确实际含义并做相应验证；
只有出现具体核心缺陷、能力阻塞或实际性能问题时，才展开针对性的深入检查。

完整历史进度和开发规范已放入[归档](archive/README.md)，按需查阅，
不作为下一轮应用设计的必读材料。当前没有运行中的核心重构。

字段组合与沿线诊断来自两个 `gpt-6-astra`、`high` 独立 worktree 任务，
已按用户要求汇总到当前分支，原分支及工作区保留。范围见[并行任务记录](parallel-work.md)。
早期大数据探索见[内存峰值与实现路径](memory-workflows.md)。在 `0bcbe2e` 上已完成
[读取、字段选择与结果交付优化](selected-delivery-optimization.md)：最终存储直读、
有效 halo 入口检查、受限 Source 与缓存、选分量采样、调用者输出和轨迹/剖面分片交付。
[调用说明](../selected-delivery.md)列出每个入口的最低支持要求。
原始叶块可合并归约及热 LOS/QSL 可恢复积分状态仍保留为后续候选。

后续按具体数据和应用目标选择方法、步长与扰动间距，并检查收敛；
不预先引入新的分析会话框架、统一执行器或核心重构任务。

应用层已完成一轮 simplify 与分析能力复查，见[修复结果和后续建议](application-review.md)。
其中积分、MHD 恢复、结果文件、字段组合和沿线诊断均已完成整合；
剖面保存/恢复及按种子分片交付也已提供。时间序列及其余补强事项继续保留待讨论。
