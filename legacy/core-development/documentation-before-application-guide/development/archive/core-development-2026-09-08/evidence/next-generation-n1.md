# 新一代 simesh N1：独立来源、准备与直接消费

日期：2026-09-08。N1 已完成。本页面向开发协作。

实现位于仓库根目录 `analysis-core/`，独立包版本 `0.2.0.dev0`。采用资产固定于
`b91bbc015d882ffd7dfcd7e10590cdce559f8532`；来源 worktree 和原主包未修改。
本次终点是[设计](../next-generation-design.md)中的 N1，不包括 N2 全域
coordinate-phase 重构或 N3 的完整 twist/LOS 接入。

## 实际实现的边界

- `Source` 只拥有字段目录、读取及来源寿命，不再绑定 `fill`、鬼点 scheme、
  支撑容量、准备工作区或缓存。文件打开只建立索引和几何；字段在请求时选择，
  原始文件字段位置保持稳定。
- `Mesh` 保存整数森林、原始叶身份和共享物理几何。区域采用相交完整叶块，
  请求框、实际目标叶和额外支撑分别表达；未构造裁剪后的伪森林。
- 区域准备分为几何绑定、来源读取、传递执行和成功发布。保留最新 SAME/FINER
  编译批量核，以及已验证的粗层支撑/minmod/物理扩宽算法。后者仍有旧 Python
  编排，本轮没有把它宣称为全新实现。
- `Fields.storage_halo` 表示存储偏移，`valid_halo` 表示实际有效层数。
  独立 backing 在源关闭后可用，准备 scratch 在发布时可释放。导数通过明确的
  整数输入偏移消费有效区域，不以分配形状推断有效性。
- 采样、curl/导数、切片和普通磁力线直接消费 Fields，保持原算术和接受前缀。
  线程池按独立目标划分，失败时等待全部工作者结束，不提前释放借用输入。
  全部追踪结果写入预分配输出，避免再做一份完整 concatenate。
- `iter_prepared` 复用有界工作区和最终批缓冲；借用在迭代推进/关闭时结束。
  这提供最小分批寿命边界，不建立 LRU 或让直接消费者自动读取。
- 新包未引入 AMRMesh、旧 Dataset 或运行时 `simesh_rewrite`。来源读取包含
  已采用的全域批量解码资产；N1 不宣称全域准备或旧用户工作流已经迁完。

资产来源见 [ASSETS.md](../../../../../../../ASSETS.md)。54 个原语/格式
文件核对为仅命名空间替换；选中的定位、插值和 RK 数值代码保持原文。
导数 kernel 新增整数偏移参数，浮点表达式和求和次序保持不变。新来源、字段、
准备和直接消费者的组织代码单独实现，不把文件改名计作算法优化。

## 核心检查

`analysis-core/tests/test_native_workflow.py` 共 9 项参数化检查：

- 混合层级非线性场的乱序字段/叶读取，完整区域准备与全域同 scheme 对照，
  跨选区的真实支撑，以及源关闭后的保留数组和采样。
- 仿射场的解析 curl；在 backing 外面增加 NaN 层后，仍按独立有效范围求导
  和采样；收缩有效范围后不再读取无效层。派生结果与主场不共享值存储。
- 分批借用期内稳定、推进/关闭后失效，准备失败不发布，以及不足预算拒绝。
- 常量场解析追踪、1/4 工作者与不同批次的逐值一致、接受前缀、域外种子、
  缺失覆盖及空输出。
- 大小端、普通/带 staggered 尾部、不同保存 ghost 范围的文件到准备场组合；
  原始浮点位模式、文件改变与截断尾部拒绝。

普通构建 9 项通过，OpenMP 构建 9 项通过，最终恢复普通构建后 9 项通过。
检查显式确认 OpenMP 从 enabled=True 恢复为 False。N1 公开消费者仍使用
线程池；这不构成新的公开 OpenMP 调度入口或性能声明。

## WENO 同范围比较

原文件为主目录 `data/weno509_sub_0000.dat`，1,045,232,320 字节、22,614
个叶块。每次在独立解释器中只加载一个版本，双方使用相同 Python 3.11.14、
NumPy 2.4.6 和 Cython 3.3.0；不同时进行构建或其他本任务的计算负载。

冻结请求：三分量 B，物理域各轴 0.40–0.60 的区域，实际 681 个完整目标叶；
exact-phase、两层 halo、连续边界、128 支撑容量。额外核对三个乱序原始叶的
b3/b1 读取；计算完整区域 curl、48×40 切片、64 个采样点，以及 64 条最多
64 步、保留轨迹的磁力线。消费使用四工作者，准备仍串行。

所有追踪共 4096 个接受步。准备产生 4048 次支撑叶装载、49,741,824 字节
逻辑字段读取。17 组完整数组逐值比较通过，各次重复输出摘要也一致，包含
主场、全部有效 curl、像素、所有者/有效性、接受位置、步数、终止及轨迹。
摘要计算和完整数组比较在正式计时之外。

### 计时边界修正

初轮 `n1-weno/` 从包导入之后计时，完整读取/准备/消费中位数为
0.2080 / 0.1790 秒。检查发现两代加载读取/准备模块的时机不同：新包有更多
导入在这个起点之前发生。因此不把该差值或约 14% 当作整体加速结论。
原始结果继续保留，首轮旧版 11.8362 秒（其中打开 10.8426 秒）的异常也未删除。

修正轮 `n1-weno-startup/` 在共同的 NumPy 就绪状态下，从加载 simesh 开始，
直到科学结果全部完成。每轮新解释器、交错顺序，首轮单列，之后三次取中位数。
不包含进程启动、共同 NumPy 导入或计时外的结果核对；操作系统缓存未控制。

| 秒；各自同轮中位数 | 固定来源 | 新 N1 |
| --- | ---: | ---: |
| 包/API 加载 | 0.01061 | 0.03450 |
| 打开与来源建立 | 0.07387 | 0.03374 |
| 原始字段检查及准备 | 0.13806 | 0.14072 |
| 后续消费者 | 0.00716 | 0.00642 |
| 包加载后文件到结果 | 0.22000 | 0.18089 |
| 包加载至结果的完整计时 | 0.23401 | 0.21539 |

完整行由每次实际起止时间计算，不是各阶段中位数之和。原版本重复完整范围
0.22466–0.23632 秒，新版为 0.21383–0.42993 秒。新版较慢的一次主要发生
在读取（0.2444 秒，对照其余新版约 0.031 秒），鬼点约 0.089 秒保持相近；
该次有 3 次主要缺页，但不能仅凭这一数量归因全部暂停。首轮旧/新完整计时
为 0.73292 / 0.25185 秒，均保留其实际缺页信息。

本轮结论是完整结果一致，未发现稳定的计算退化；包加载边界修正后仍有较小
中位数差异，但存在读取波动，不宣称普遍或稳定加速，也不将收益归给未改的
鬼点数值算法。后续全域、长追踪与 LOS 性能属于 N2/N3 的实际工作流验收。

## 独立构建和安装

`analysis-core/pyproject.toml`、`setup.py` 和 Makefile 只发现本目录源码。
原语保持自身的检查/除法编译语义；OpenMP 和优化标志单独应用于分析核。
`make build` 强制重编译，避免切换 OpenMP 标志后误用旧构建产物。

最终从源码 sdist 构建普通 wheel，再解压到独立安装目录，以 `python -I -S`
禁用用户路径及 editable 钩子。仅加入安装目录、第三方依赖和测试目录后，
9 项检查通过；逐个已加载 simesh 模块确认路径属于该安装，未加载 rewrite。
安装包再次直接读取 WENO，完成相同工作流，17 组输出与固定来源一致。
这次安装后首次调用的计时没有拼入正式性能组。

最终产物位于 `analysis-core/benchmark-results/n1-final-package/`：

- 源码归档 `simesh-0.2.0.dev0.tar.gz`，4,006,247 字节，SHA-256：
  `ec6dbe5eac2b9b4161c2e6274999d3b32459f014e1f6e9cb2d4edb34dcd39d84`。
- wheel `simesh-0.2.0.dev0-cp311-cp311-macosx_11_0_arm64.whl`，5,689,750 字节，
  SHA-256：`63a59cf76dc9023fc8b6260a0b05211b738f5a04ca04f1faf1a10e37638bbc1b`。
- `install-verification.json`、`installed-weno.json` 与 `installed/` 保留安装验收。

## 资源、命令和剩余工作

本机 8 GiB 内存、8 逻辑 CPU。计算检查最多四工作者，准备受控数组上界
75,341,801 字节；修正比较组的进程峰值 RSS 最大 110,510,080 字节。
控制数组、进程 RSS 和文件页缓存是不同指标。新环境、构建、源码及证据共
不足 0.5 GiB，低于 2 GiB 新增暂存额度，磁盘仍有约 30 GiB 可用。

复现从 `analysis-core/` 开始，比较输出目录需使用新名称：

```bash
make build
make test
.venv/bin/python scripts/run_comparison.py \
  --file ../data/weno509_sub_0000.dat \
  --donor /Users/astery/.codex/worktrees/d538/simesh \
  --output-dir benchmark-results/new-n1-comparison --repeats 4
SIMESH_OPENMP=1 make build
OMP_NUM_THREADS=1 make test
SIMESH_OPENMP=0 make build
make test
.venv/bin/python setup.py sdist --dist-dir benchmark-results/new-package
.venv/bin/python -m pip wheel benchmark-results/new-package/simesh-0.2.0.dev0.tar.gz \
  --no-deps --no-build-isolation --wheel-dir benchmark-results/new-package
.venv/bin/python scripts/verify_install.py \
  --wheel benchmark-results/new-package/simesh-0.2.0.dev0-cp311-cp311-macosx_11_0_arm64.whl \
  --target benchmark-results/new-package/installed
```

源码和安装检查采用与固定来源相同的工具版本；正式日志在新目录的
`benchmark-n1.log`、`benchmark-startup-n1.log`、`build-n1-*.log`、
`package-n1-final.log`，原始 JSON/完整数组在对应 `benchmark-results/` 子目录。

N1 可用入口见[新包 README](../../../../../../../README.md)。下一步是 N2：
全域 coordinate-phase 准备与最终值/工作区所有权拆分，同时保留快速读取和
最新局部线程池阶段。当前粗层准备编排、E5 计划执行、twist/LOS、有界追踪、
旧 Dataset/写回/2D 支持尚未全部迁入；E1 重分块仍是探索资产。真实大输入、
新科学能力与最终旧包替换不能由 N1 的完成状态推断。
