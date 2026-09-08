# Analysis-core 当前进度

更新：2026-09-08。继续开发前参阅本页和 [开发流程](development.md)。

## 当前实施：两个独立采样复用探索

用户已授权启动固定 N4 基线上的两个独立任务/worktree，分别研究 B/curl 联合
采样和热 LOS 同单元多节点复用。实现、完整前后效率比较、独立评审与整合
验证属于本轮端点，详见[共同启动约定](explorations/sampling-reuse.md)及其两份任务说明。
当前协调任务保存基线并串行安排编译/测量；探索任务不修改父目录旧实现。
重分块、自适应积分、新重构、读取改造及其他候选暂不进入本轮。

## 当前目标：在 analysis-core 内建设独立的新一代 simesh

用户已确认独立建设目标，并要求纳入各 worktree 的最新策略实现和既有
analysis-core 探索。新实现位置是仓库根目录 `analysis-core/`；详细范围、
固定来源、探索处置和继承缺口见 [目标与资产继承](next-generation.md)。
此前“不另建包、继续完善现有 simesh.analysis”的范围已被本次目标替代。

最新实现来源为 `d538` worktree 的 `b91bbc0`，已经包含 `85cb` 的结构整合、
后端优化、线程池统一及 `b403` 的效率收敛成果；主目录不是最新运行基线。
各来源和主目录原有未提交成果继续保留。独立目录已经从目标/资产导航发展为
下述 N1–N4 可运行实现；其状态和验收不能与旧主目录或来源版本混淆。

首版[模块、接口与实现取舍](next-generation-design.md)已完成源码核对和设计。
选择 Source 与准备分离、存储偏移与有效范围分离、显式数值 scheme；新包在
`analysis-core/src/simesh/` 独立构建。E1–E5/T01–T18 已映射到相关职责，
AMRMesh 拆分、旧粗层/计划编排和旧功能迁移均有明确处置。N1–N4 支持范围以
新包 README 和 MIGRATION.md 为准，后续科学/规模目标另行推进。

用户要求的下一实施步 N1 已完成：[实现与验收证据](evidence/next-generation-n1.md)。
根目录 `analysis-core/` 现有独立 `simesh 0.2.0.dev0`：文件/数组 Source、
共享整数 Mesh、区域 exact-phase 准备、独立存储/有效层描述、最小分批借用，
以及采样、curl/切片和普通磁力线。新环境与构建均在此目录，原包和来源未修改。

普通/OpenMP/恢复普通构建及最终隔离 wheel 各通过 9 项核心检查。WENO 681 个
目标叶、64×64 步追踪的 17 组完整输出与固定来源一致。修正导入计时边界后，
含包加载的完整中位数为 0.2340 / 0.2154 秒，但有读取暂停，不宣称稳定加速。
N1 的普通构建源码包与 wheel 保留在
`analysis-core/benchmark-results/n1-final-package/`，调用见[新包 README](../../analysis-core/README.md)。

本机为 8 GiB 内存、8 逻辑 CPU，沿用至多四计算工作者、2 GiB 受控活跃数组、
2 GiB 新增暂存及至少 2 GiB 可用磁盘。N1 比较峰值 RSS 约 105 MiB，新增产物
不足 0.5 GiB；原样本只读。N2–N4 的最新交付见下节。

## 已完成：N2 全域准备与独立所有权

N2 已完成，[实现、缺陷修正和验收证据](evidence/next-generation-n2.md)记录完整范围。
固定来源仍为 `b91bbc0`，N1 的已验收包和结果保留。新包现已支持 full-domain
coordinate-phase、全域批量读取、独立最终数组与 coarse 工作区，以及局部
阶段线程池和显式 OpenMP。未覆盖全域的请求拒绝，不扩域或切换数值方案。

实现从共享整数森林生成 canonical 邻接/坐标，用临时编译绑定借用 NumPy
最终数组和 coarse 缓冲，没有带入旧 AMRMesh/AMRForest 或其分配所有权。
原始值与 coarse 错开存活，发布没有完整复制。WENO 全部邻接/坐标、小混合场
和全域准备/curl/切片/追踪的 15 组完整输出均完成对照；两版大数组不同时驻留。

本机复核仍为 8 GiB 内存、8 CPU、约 30 GiB 可用磁盘；计算至多四工作者、
活跃数组不超过 2 GiB，N2 新增暂存不超过 2 GiB，保留至少 2 GiB 可用空间。
源文件/来源 worktree 只读。普通构建 15 项通过、1 项 OpenMP 专用检查跳过，
OpenMP 构建 16 项通过；当前已恢复普通构建。最终源码包、wheel 和隔离安装
证据在 `analysis-core/benchmark-results/n2/package/`，安装后全域 WENO 结果一致。

小场对照定位并在新代修正旧 prolong 支撑表的越界：双侧物理边界会索引
不存在的第 4 行，旧/初始抽取常量场出现 3456 个零值。修正后常量保持；
新代数值标识为 `canonical-coordinatephase-cont-v2`，原方案算术及不触发
缺陷的案例继续作逐值比较。另在几何边界拒绝不可表示的延拓模板索引。

WENO 准备受控上界由约 1.537 GB 降至 1.238 GB；整体 RSS 仍由大 B/curl
同时保留所限制。最终四工作者、含包加载的同轮完整中位数为旧版 1.278 秒、
新版 1.171 秒，保留全部波动记录，不宣称固定普遍倍率。

N3 已接入长追踪/twist、标量/热 LOS、可选几何计划、有界消费和大输出，见下节。
区域粗层编排仍有旧实现；N4 已接入 Dataset/写回兼容层。父目录旧路径保持
原样，新代 v2 和兼容副本已修正该已知越界。科学范围以设计和各自验收为准。

## 已完成：N3 科学消费者、计划与有界交付

用户要求的 N3 已在独立目录完成，[科学、寿命与完整工作流证据](evidence/next-generation-n3.md)
记录实际范围。来源保持固定 `b91bbc0`，N1/N2 的源码包、wheel 与比较结果保留。
新包已接入接受段 twist/重追踪、标量与历史 AIA171 热 LOS、显式几何计划、
准备池和有界计算、独立文件任务及流式均匀输出。普通消费者只接收 Fields；
补数留在 `bounded` 协调入口，派生结果的逻辑身份不保留主场数值。

E5 计划与 Source/数值寿命分离，每次绑定值和限制器。可选原始值缓存关闭后
释放 backing，父源失效也禁止命中读取；没有重启重分块或已回退缓存策略。
热场转换把不变绑定移出每叶循环，保持历史公式、单位和两种重构次序。

WENO 65,536 种子共 21,573,979 接受步、twist/重追踪、500² 双视角热图、
有界计划/LOS/流式组合和独立进程全域 curl 的完整记录数组均与固定来源一致。
温度是明确构造输入，不宣称由快照恢复或完成物理标定。同轮完整中位数旧/新为
长追踪 3.182/3.037 秒、有界组合 2.928/2.755 秒、最终热图 8.239/8.337 秒；
保留首用及波动，不宣称普遍倍率。初次热图回归及修正后的复测均保留。

普通构建 21 项通过、2 项 OpenMP 专用检查跳过；OpenMP 构建 23 项通过，
并完成真实 WENO 追踪、64² 热图、有界流程与普通构建的完整数组核对。
最终恢复普通构建；N3 源码包、wheel、隔离安装和安装后真实流程证据保存在
`analysis-core/benchmark-results/n3/package/`，无需父源码或来源 worktree。

实际百万种子运行完成 3200 万接受步；1000³ 网格按拥有结果的切片累计交付
24 GB 数值，单片最大 33 MB，进程峰值约 338 MiB。没有写出完整体积文件，
也没有据此宣称真实 10–20 GB 或超内存输入已经验收。

本轮至多四计算工作者、2 GiB 受控活跃数组、N3 新增暂存不超过 2 GiB、
至少 2 GiB 可用磁盘。各大数组工作流顺序运行，原输入与来源只读。
N4 兼容交付已完成，见下节；2D/周期/CT 等旧支持面仍按实际能力区分，
不直接归为新原生分析已支持。
下文属于前序交接与历史成果，其任务位置、待办和范围不覆盖本节。

## 已完成：N4 兼容迁移与首版独立交付

用户要求的 N4 已在根目录 `analysis-core/` 完成，来源固定 `b91bbc0`。
[兼容交付证据](evidence/next-generation-n4.md)记录具体保留/修正范围；公开
调用和旧分析接口映射见 [MIGRATION.md](../../analysis-core/MIGRATION.md)。
N1–N4 首版建设已闭环，父包、来源与 N1–N3 的源码/wheel 继续保留。

独立安装已包含八个公开 AMRVAC 入口、mutable Dataset、命名派生/导数、
普通字段读写、数组建文件、2D singleton-z、level-1 VTK、potential-field
和数组配置工具。兼容模块显式使用时才加载，新 Source/Fields/科学消费不依赖
旧 AMRMesh。`source_from_dataset` 从已加载列复制独立 Source；
`write_amrvac` 验证完整覆盖和显式元数据，按原始叶顺序复制内部值并原子发布。

旧 periodic 只有文件元数据，没有周期 ghost 邻接。新兼容保留元数据往返，
拒绝周期 ghost；拒绝没有 CT 载荷却保留 staggered 标记的普通写出。新原生
字段导出明确是普通值产品，不承诺 solver restart。兼容 prolong 表沿用 N2
已知越界修正；VTK 保留原端点式坐标，不静默更改已公开行为。

固定来源的非重型兼容检查 64 项通过；缺少 SciPy 的 1 项在安装可选 SciPy 后
单独通过。新完整普通构建 26 项通过、2 项跳过；OpenMP 构建 28 项通过，
并验证原生三模块启用、兼容 AMR 保持串行。最终恢复普通构建，源码→wheel→
隔离安装及安装后的实际科学/写出流程通过。产物在 `benchmark-results/n4/package/`。

WENO 完整导出/重读共四对交错运行，b3/rho 全部数值及 186,442,976 字节
输出文件逐次一致。完整中位数旧/新约 2.972/2.006 秒，主要收益来自既有
批量读取；写出约 1.128/1.156 秒，不宣称新序列化算法加速。峰值 RSS 旧/新
约 622/463 MB，输入与来源只读，临时导出验证后删除。

本机 8 GiB/8 CPU、约 32 GiB 可用盘；沿用最多四工作者、2 GiB 受控活跃
数组、N4 新增暂存不超过 2 GiB、至少 2 GiB 可用空间，各大工作流顺序执行。
兼容 API 不实现 native memory_limit，受其自身数组模型约束；本轮资源通过
具体工作量和实际峰值核对，不能把原生 admission 宣称覆盖所有旧对象。

本次实施端点完成，没有自动开始新的科学或规模扩展。后续仍有真实 10–20 GB/
超内存输入、周期/2D/CT 原生分析、有界非线性热 LOS、精确足点/Q/GPU，及
尚未采用的 E1–E5 探索。旧均匀网格 OpenMP 和 VTK 坐标变更仍是独立议题。

## 先前整合范围与统一状态

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
