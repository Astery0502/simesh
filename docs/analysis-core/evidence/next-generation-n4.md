# N4：兼容迁移、数据边界与独立交付

日期：2026-09-08。面向开发协作。实现位置为仓库根目录 `analysis-core/`，
固定来源仍为 `d538` worktree 的 `b91bbc015d882ffd7dfcd7e10590cdce559f8532`。
N3 源码包、wheel、全部数值结果和父目录实现均保留。

## 实际迁移方式

本轮采用设计允许的自包含兼容实现：保留八个 `simesh.amrvac` 公开入口，
以及 Dataset 的字段注册/物化、导数、选择和删除；写回、数组建文件、level-1
VTK 和 2D singleton-z 流程随其必要依赖一起进入独立安装包。

兼容层包含 9 个 AMRVAC Python 文件、3 个 AMR 扩展及其头文件、少量包初始化
和运行状态函数。`simesh.tools.potential_field_green` 与
`simesh.utils.configurations` 作为独立数组工具保留。来源清单与改动见
[ASSETS.md](../../../analysis-core/ASSETS.md)，用户选择入口、旧分析调用迁移、
布局和限制见 [MIGRATION.md](../../../analysis-core/MIGRATION.md)。

这不是把旧 AMRMesh 重新放回新数据模型。只导入 `simesh` 并执行原生 Source/
Fields/科学消费者时，不加载 `simesh.amrvac` 或 `simesh.utils`；兼容模块在
显式使用时加载。其旧 mutable 所有权只存在于兼容 Dataset 中。
新准备和科学消费者继续使用 N1–N3 的独立 Mesh、Fields 和已验收编译内核。

新增两个显式数据边界，代码位于 `io/products.py`：

| 接口 | 行为与占用 |
| --- | --- |
| `source_from_dataset` | 要求已加载的非周期 3D Dataset；按已加载名称选择原始或物化派生列，复制一次内部值并重建独立整数几何。后续 Dataset 编辑、删除和回收不改变 Source。没有带入 ghost、回调 recipe 或旧 Mesh 引用。 |
| `write_amrvac` | 要求完整原始叶覆盖及匹配的 v5 元数据。按 slot 映射恢复原始 SFC 顺序，显式分配一份 field-major 内部值，沿用规范 serializer 写普通字段。两层 halo、单位标签、scheme 和派生身份不冒充文件字段或物理标定。 |

导出拒绝局部覆盖、几何不匹配和不可表示的文件坐标。字段名按文件格式校验；
tree/计数/偏移与所导出字段重建，时间和模型元数据由调用者明确提供。
导出的 curl 等分析数据不等于可供求解器恢复的完整物理状态。

新导出先写同目录临时文件，成功后发布。默认拒绝已存在或同时创建的目标；
明确 `overwrite=True` 时只在完整序列化后替换。写入失败清理临时文件并保留
原目标。旧兼容 writer 的整体事务语义没有被宣称同步升级。

## 沿用行为和明确处置

### 非周期边界与已知越界

兼容 `cont/symm/asymm/noinflow` 参数与所需法向速度字段保持原行为。
已知双物理边界 prolong 表仅有三行的问题在本目录按 N2 v2 同样修正：
增加第四行并组合低/高侧扩宽。9 叶混合级别常量场的全部 padded 值为 1，
中心差分及其写出结果为 0。没有修改父目录或来源版本。

### 周期元数据与 CT

源码核对确认：旧 Dataset 保存 `periodic`，但构建森林和 ghost 邻接时没有
传入周期条件。因此文件元数据往返与周期 ghost 计算是两种能力。
本轮保留前者，明确拒绝周期文件的 `ghost_width>0`；原生 Source 继续拒绝周期。
没有声称已经接入周期分析算法。

旧规范 writer 总是只写普通值，却可能保留 `staggered=True` 的头部，导致输出
宣称存在实际未写入的 CT 数据。新兼容 writer 在写入前拒绝这种请求。
原生 Fields 导出则明确是普通值产品，写出 `staggered=False`。WENO 比较也在
两版均明确只导出普通字段的条件下进行，不把缺失 CT 的旧错误文件当作参考。

### 兼容均匀采样与 VTK

保留旧 zero/linear、`uniform_full()` 和数组布局的公开语义。VTK 继续使用
历史测试要求的端点式 structured-points 坐标，原有数值与字段顺序不变；
文档明确它不是新原生 cell-center 采样坐标。没有静默更改这项已有约定。

### 编译与数组工具

兼容 AMR 保留原 Cython 数值指令，但固定普通串行编译。原因是来源
`evidence/backend-efficiency.md` 已在旧版本复现 `uniform_grid_linear`
OpenMP 共享边界的 52 项输出差异；本轮不重新宣称这条旧并行路径已验收。
`SIMESH_OPENMP=1` 仍可启用新原生 preparation/tracing/LOS，兼容 AMR 则继续
报告 `enabled=False`，二者不混为同一构建状态。

potential-field 的 NumPy 直接算法保持原文；SciPy FFT 为可选 `fft` extra。
安装 SciPy 1.17.1 后补做已有 FFT/直接数值对照。数组配置公式只按固定来源
保留，没有为迁入改名重造一组公式测试或新增物理准确性声明。

## 验收

新增 5 组组合检查覆盖：Dataset 编辑/物化/移除及 Source 独立寿命、乱序叶
派生导出和完整数值重读、失败不破坏已有文件、2D/周期元数据/VTK 往返、
双边界常量修正和派生写回，以及独立数组工具到 Dataset 的完整转换。

在独立新解释器中直接运行固定来源的非重型 Dataset、write 与 potential-field
检查：**64 通过、1 项缺少 SciPy 跳过、3 项重型取消选择**。随后安装可选 SciPy，
原有 FFT/直接对照 **1 项通过**。这些检查的 simesh 模块均来自新目录，来源
仅提供测试函数，没有混装或借用运行时代码。

本地完整普通构建为 **26 通过、2 项 OpenMP 专用检查跳过**。
OpenMP 构建为 **28 通过**，并检查三个原生模块启用 OpenMP、兼容 AMR 未启用。
最终恢复普通构建，从独立源码包重建 wheel，隔离安装通过同样的普通检查。
隔离解释器禁用 site/editable hooks，逐模块验证运行路径属于安装目录；安装后
再次执行真实 WENO 的普通字段导出和科学消费者完整数组比较。
最终 wheel 为 7,302,257 字节，隔离安装为 26 通过、2 跳过。安装后导出文件
与完整数值哈希一致，65536 种子磁力线/twist/重追踪的 27 组数组也与 N3 一致。
结果见 `package/install-verification.json` 和 `package/workflow-verification.json`。

日志/产物均在新目录：

- `test-n4-compatibility-fixed.log`、`test-n4-retained-suite.log`、`test-n4-fft-reference.log`；
- `build-n4.log`、`test-n4-normal.log`、`build-n4-openmp.log`、`test-n4-openmp.log`；
- `build-n4-restored.log`、`verify-n4-package.log`；
- `benchmark-results/n4/io/`、`benchmark-results/n4/openmp-build.json`；
- 最终普通构建、源码包、wheel、安装和真实流程记录见 `benchmark-results/n4/package/`。

## WENO 文件到文件完整对照

原输入 `data/weno509_sub_0000.dat` 只读，1,045,232,320 字节、22,614 叶。
选择原始顺序外的 `b3/rho` 两字段，明确生成普通值文件，重新打开并读取所有
值。输出每次为 **186,442,976 字节**；全部数组哈希和整个文件哈希逐次一致。
计算完比较后删除本轮临时导出文件，只保留 JSON。

脚本 `scripts/compare_n4.py` 用 `-I -S` 在一个进程只加载一个版本；共四对
交错运行，保存首对及后三对中位数。计时包含包导入、输入读取、值/布局转换、
写出、发布和重开；预期值哈希时间单独记录并排除，最后数组和文件哈希也在
计时之外。未控制操作系统文件缓存。

| 环节 | 固定来源中位数，秒 | 新实现中位数，秒 |
| --- | ---: | ---: |
| 包导入 | 0.0170 | 0.0744 |
| 打开并读取 | 1.4888 | 0.4296 |
| 导出 | 1.1275 | 1.1561 |
| 重开和读取 | 0.3204 | 0.3251 |
| 完整流程 | 2.9717 | 2.0056 |

分环节中位数不能保证相加等于完整中位数。原始记录保留在 `n4/io/summary.json`。
主要差异来自已有原生批量读取；序列化代码保留，不把完整收益宣称为新 writer
算法的加速。四轮峰值 RSS 旧/新分别为 621,707,264 / 462,749,696 字节。

机器仍为 8 GiB、8 逻辑 CPU；最多四计算工作者、2 GiB 受控活跃数组，
各大文件流程顺序执行。N4 暂存保持低于 2 GiB，磁盘可用约 32 GiB，原输入
和来源 worktree 没有更改。兼容 APIs 不实现新 memory_limit；本轮旧对照的
数组和产物由具体工作量控制，原生边界则显式 admission。
保留的 N4 测量、源码包、wheel 和解出的安装内容合计约 55 MiB；一次临时
普通字段文件约 186 MB，验证后删除。

## 交付终点与后续范围

N1–N4 的首版独立建设和兼容交付已闭环：新环境可直接运行原生分析与选定的
旧公开工作流，原版本和已验收产物可回退。不把历史内部 `legacy`/rewrite
实现全部搬入安装包，也不将接口连续性等同于所有长期科学目标已经完成。

后续仍可按独立目标推进周期/2D/CT 原生分析、有限内存非线性热射线、真实
10–20 GB/超内存输入、精确足点/Q、GPU，以及 E1–E5 尚未采用的探索；本轮
没有自动启动它们。兼容 VTK 坐标和旧均匀网格 OpenMP 应在明确的后续变更中
处置，不能由 N4 完成状态推断已经更换算法或修复。
