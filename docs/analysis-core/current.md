# Analysis-core 当前进度

更新：2026-09-08。继续开发前参阅本页和 [开发流程](development.md)。

## 本轮范围与统一状态

用户要求将辐射与优化探索 worktree 的实现统一回主工作目录，更新工作流记录，
然后移除该 worktree。本轮负责整合与兼容性验证，不扩展新的数值策略或性能探索。

统一目标为 `/Users/astery/science/simesh`，分支
`codex/analysis-core-p0-p4`。两条工作线互补：主分支保留数值缓存、调度、
多视角标量 LOS 和独立进程全域 curl；`codex/analysis-core-euv-geometry`
提供 AIA171 热辐射、编译射线遍历、可选几何准备计划与重分块可行性证据。
之后以本页作为唯一恢复入口，不再将这些实现分配给另一个独立会话。

主目录原有未提交文档已保存为 `9fa1bc3`，其中包括
[准备与存储设计评估](preparation-reuse-review.md)。保留该评估的范围：优先准备一次、
反复使用，同时支持有限内存处理；最终存储与准备工作区的独立生命周期仍是后续提案，
本次整合不将它标为已实现，也不重新启动重分块探索。

## 已交付功能与证据

| 工作线 | 实际状态 |
| --- | --- |
| P0/P1 | 原生字段、两层主场 halo、稳定有限容量借用与直接消费已交付；[P1 证据](evidence/p1-native-fields.md) |
| P2 | 接受步前缀语义、并行磁力线、终止信息和有限缓存比较已交付；[P2 证据](evidence/p2-field-lines.md) |
| P3-D | 全域 curl(B)、独立保留的派生场与轴向/斜切片已交付；[D 证据](evidence/p3-d-global-slices.md) |
| P3-F | 所选接受段 twist、可选轨迹与显式重追踪已交付；[F 证据](evidence/p3-f-twist.md) |
| P3-L 标量 | 全域标量积分、WENO 密度柱与多视角调度已交付；[标量证据](evidence/p3-l-scalar-los.md) |
| P3-L 热辐射 | 首个模型已获委托并落实：历史 AIA171 表、显式 H/He EOS、密度/长度单位、外部 K 温度或等温输入，以及两种重构顺序；[热辐射证据](evidence/p3-l-thermal.md) |
| 热射线优化 | AMR 树区间遍历、共享内联插值、编译响应与 1/2/4 工作者已交付；保留 Python 参考路径，默认一个工作者；[射线证据](evidence/thermal-rays.md) |
| E5 几何计划 | 可选 `build_fill_plan` / `FillPlan.prepare` 已采用；几何可复用，数值和限制器每次重新计算；[计划证据](evidence/geometry-plans.md) |
| E1 重分块 | 几何与映射原型已完成，暂不替换生产叶块布局：稀疏请求会扩大准备范围；[重分块证据](evidence/rebricking.md) |
| 运行时优化 | 可选原始值缓存、相近视角分块复用、独立进程全域 curl 与显式线程/OpenMP 调度已交付；未采用的命中追踪与额外 halo 暂存保持回退状态；[运行时决策](runtime-execution.md)、[证据](evidence/runtime-execution.md) |
| P4 | 安装、兼容性、百万种子和实际 1000³ 输出检查已完成；真实 10–20 GB / 超内存输入仍未验证；[集成](evidence/p4-integration.md)、[规模](evidence/p4-scale.md) |

## 整合边界与验证

- `FieldSource` 同时保留 `validate_values` 和 `plan_builder`；计划读数使用
  同一个可选原始值缓存，数据源关闭或改变后不得绕过生命周期检查。
- 计划统计中 `read_value_bytes` 表示底层实际读取，`requested_value_bytes`
  包含缓存命中；加入文件计划、重复读取、字段选择及失效检查的组合回归。
- 编译热射线与原生消费者共享 `native.pxd` 插值实现，必须一起重建。
  保留当前原生模块的 OpenMP 功能；热射线仍使用自身的驻留字段线程池调度。
- `FillPlan.prepare` 显式生成独立字段，不自动接管 `PreparedPool` 缺失调度。
  几何计划不保存缓存槽、字段值、代际或 minmod 结果。
- 默认构建完整 analysis 测试：30 项通过；新增测试覆盖几何计划与原始值缓存
  共存、字段重绑定、文件修改/关闭拒绝和脱离源后的结果读取。
- `make test PYTHON=.venv/bin/python` 通过；仓库默认未启用的重型用例保持跳过。
- 干净源码构建的 wheel 为 9,609,216 字节；通过 `python -S` 隔离检查，
  验证缓存文件源、追踪、进程 curl、多视角 LOS、几何计划和并行热射线/参考一致性。
- OpenMP 构建下 `test_execution` 与 `test_thermal` 共 5 项通过，
  包括静态/动态调度以及 1/2/4 工作者的一致性。
- 已恢复默认非 OpenMP 构建，并明确检查 `enabled=False`；恢复后相同 5 项检查通过。
- 已通过 `git worktree remove` 移除 `/Users/astery/science/simesh-euv-geometry`；
  仅保留主工作目录。原分支指针继续保留，提交历史和迁回的原始产物可追溯。

本轮命令记录：

```sh
make build PYTHON=.venv/bin/python
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -v
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.validate_install
make test PYTHON=.venv/bin/python
.venv/bin/python build.py --inplace --group analysis --openmp
OMP_WAIT_POLICY=PASSIVE OMP_DYNAMIC=FALSE PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest test_execution test_thermal -v
SIMESH_OPENMP=0 .venv/bin/python build.py --inplace --group analysis
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest test_execution test_thermal -v
```

日志位于 `benchmark-results/analysis-core/integration-*.log`；隔离源码、wheel 和
运行目录为 `benchmark-results/analysis-core/source-build-1uugwyhx/`（约 177 MiB）。
本轮未重新运行大型性能矩阵，没有新增性能加速声明。

历史测量继续保留其原范围，不将它们宣称为本次重新测量：

- 热射线：三个 64×64 视角与 Python 参考的最大绝对差不超过
  `2.73e-12 DN/s/pixel`；实际 500×500 图像在 1/2/4 工作者下逐值一致。
  500×500 查询中位数分别为轴向 23.00/8.40/18.27 秒、斜向
  29.41/15.59/9.22 秒、对角向 17.86/12.09/6.41 秒。
  当时存在明显内存压力，不据此宣称四工作者普遍最优。
- 几何计划：所声明的稀疏/稠密重复准备组合中测得 2.56/6.67 倍改善；
  构建成本、保留内存和适用范围见原证据。
- 全域 curl 加两切片：历史单工作者约 31–35 秒，两个进程约 17.5 秒，
  四个进程约 13 秒，完整结果一致；控制数组上界约 584/735/917 MB。
- 百万种子最多 32 步：历史 1/4 工作者为 8.176/2.274 秒，共
  17,273,160 个接受步。1000³ 输出流完整消费十亿点，累计 24 GB 值，
  81.283 秒，最大 slab 33 MB；没有保存整个立方数组。

## 保留结果与可恢复提交

- 原实现检查点：`ee96824`、`f6dc48e`、`6558286`、`8a80a2f`、
  `2c00792`、`a5f1f88`；运行时采用版本 `6172bbb`，恢复记录 `cf4207d`。
- 辐射/几何原分支：继承文档 `d1aaf66`，首轮实现 `ab75a17`，
  恢复记录 `a61a0ac`，编译射线 `9f423e9`，最终实测记录 `8e35b2f`。
  通过合并保留其祖先历史，而非直接覆盖主分支。
- 原 worktree 的 28 个结果文件已逐个复制并校验 SHA-256，共
  5,823,026 字节，位于 `benchmark-results/analysis-core/euv-geometry/`。
  `migration-manifest.json` 保存清单；其中保留原始 JSON、日志、性能剖析、
  初版 C-API 探索，以及 `thermal-500-images.npz` 和
  `thermal-500-projections.png`。这些文件继续位于忽略目录。
- 主目录既有 `benchmark-results/analysis-core/` 数值产物和原始记录保持原位，
  包括全域 curl、百万种子结果和先前安装产物。

## 剩余缺口与后续入口

此前 P0–P4 已获授权，P2 是中间检查点；但整体科学与输入规模验收仍未完全结束。
首个响应/EOS/单位选择已经落实，不再保留“等待委托首个模型”的过时阻塞。

1. 真实温度与物理归一化：WENO 只有密度和磁场，没有能量/温度；历史图像使用
   制造的 0.45–1.65 MK 温度和明确的演示单位，不能当作真实快照热辐射。
   历史表缺少完整 CHIANTI/丰度/标定生成元数据，当前仪器标定仍未验证。
2. 真实大输入：仍需已有、可读的 10–20 GB Cartesian 快照。
   百万种子、十亿输出点或制造的稀疏文件不能代替该输入验收。
3. 非线性热辐射当前是驻留字段路径；有限内存响应执行、全域计划保留和生产直接
   重分块准备没有交付。它们不构成本次整合的额外必做任务。
4. 存储后续从[应用驱动评估](preparation-reuse-review.md)继续：先明确最终 C 缓冲区
   所有权，再决定准备工作区能否无完整复制地释放；不要重跑已回退的缓存探索。

当前新原生消费者仍限非周期 Cartesian 3D；不宣称新的 2D 挤出、周期追踪、CT、
非 Cartesian、精确边界足点、Q 或 GPU 支持。规范用户 API 保持默认入口。
实现和探索决策见[选定设计](native-core-design.md)，调用方式见[使用说明](usage.md)。

本机沿用 8 GiB 内存、八个逻辑 CPU 的工作配置；验证至多四个计算工作者，
单例受控数组不超过 2 GiB，新增临时磁盘不超过 2 GiB，保留至少 2 GiB 可用空间。
这些约束不保证进程 RSS 或系统页缓存上界；本次没有新增时间或 token 预算。
