# 新一代 simesh N2：全域准备与独立所有权

日期：2026-09-08。N2 已完成，本文面向开发协作。

固定来源为 `b91bbc015d882ffd7dfcd7e10590cdce559f8532`，来源 worktree
`/Users/astery/.codex/worktrees/d538/simesh` 的提交与干净状态已核对。
新实现只进入根目录 `analysis-core/`，不修改原主包或来源。N1 的源码包、
安装产物和区域证据继续保留。

## 实现与数值边界

- `prepare(..., scheme="coordinate-phase")` 支持完整源覆盖、字段选择、
  默认线程池及显式 OpenMP，默认一个工作者。未覆盖全部源叶的请求拒绝，
  不改为全域请求，也不切换 exact-phase。
- `preparation/coordinate.py` 从共享整数 Mesh 构造只含几何的邻接、坐标和
  物理边界描述。没有再构造 AMRForest 或 AMRMesh。
- `_kernels/coordinate.pyx` 的临时绑定只借用 NumPy 最终数组、独立 coarse
  缓冲和只读几何；保留局部阶段、限制/延拓算术和跨目标写入的原串行顺序。
  最终 Fields 不持有交换对象、coarse 缓冲或专用准备几何。
- 原始内部值在装入最终数组后释放，再分配 coarse，避免三份大数组同时存活。
  最终值由 NumPy 直接拥有，发布时不做完整复制。
- 全域物理槽位保留 SFC 顺序；请求的全域叶排列保存在 Selection，目录负责
  寻址。非打包顺序使用窗口访问，不为了排列叶 ID 再复制完整场。

### 修复继承的双物理边界错误

旧 prolong 支撑表第二维只有 3 行，但粗块横跨某轴两侧物理边界时，
`idphyb=2` 导致访问第 4 行。Cython 关闭边界检查后，这是越界访问。

在 `(2,1,1)` 根布局、9 叶、8³ 单元块的混合级别常量场上，固定来源产生
3456 个错误零值；初始原文抽取也完全复现。两版邻接和专用坐标逐值相同，
排除了新几何转换造成该错误。原始数组保存在
`analysis-core/benchmark-results/n2/small-donor.npz`。

新实现增加第 4 行并组合两侧支撑扩宽，全部准备值恢复为 1。新数值标识为
`canonical-coordinatephase-cont-v2`，区别于已知错误的 v1；保留原坐标算术，
不换用另一种重构。旧代码不动；不触发该条件的案例继续与 v1 逐值核对。

另在几何边界预检延拓所需的粗单元及两侧斜率支撑索引。例如原点为 1e16、
细单元间距约 1 时，舍入可能把细鬼点映射到 coarse 最后一个单元，使斜率
读取越界。新实现明确拒绝不能表示的模板，检查先于场读取和数值交换。

## 核心、几何与完整结果检查

- 普通构建：15 项通过、1 项 OpenMP 专用检查跳过。
- OpenMP 构建：16 项通过，包含显式 1/2/4 工作者的结果一致性。
- 最终恢复普通构建后再次通过 15 项、跳过 1 项，并确认 enabled=False。
- 新检查覆盖双物理边界常量保持、非立方单块边界、字段/全域叶顺序、
  独立最终数组和准备对象释放、失败不发布，以及不可表示的坐标模板拒绝。
- 额外三层、非立方块常量场的全部 63,360 个值保持为 1。
- WENO 全部 neighbor type/index/children 和 coordinate-phase 专用坐标
  与旧几何逐值一致，完整数组保存在 `n2/geometry-weno/`。
- 三层、22 叶、8×6×4 块、非零物理原点的非线性场，带保存 ghost 和
  staggered 尾部，准备、curl、两切片和追踪等 15 组完整数组与固定来源一致。
  数组及记录在 `n2/mixed-check/`，输入为 `n2/mixed.dat`。

所有权检查保留最终 NumPy 数组并删除 Fields/Source，再确认内容可读；弱引用
同时证明准备几何、邻接和 coarse 工作区已释放。最终数组的 OWNDATA 为 True，
不存在保活旧网格对象的 NumPy base 链。

## WENO 全域组合与成本

原文件 `data/weno509_sub_0000.dat` 只读，22,614 叶、普通 B 三分量。
准备包含全部 117,230,976 个 padded 值，curl 包含全部 67,842,000 个有效值。
随后完成 128² 轴向切片、96×80 斜切片、64 点采样和 64 条最多 32 步追踪，
保留轨迹；共 2048 个接受步。

每个版本在单独解释器执行，NumPy 就绪后从加载 simesh 开始计时，包括来源、
读取、准备、计算和结果生成。进程启动、共同 NumPy 导入及结果核对不在该
计时内。两版使用相同 Python 3.11.14、NumPy 2.4.6、Cython 3.3.0；操作系统
缓存未控制。比较时不并行运行本任务的构建或其他大数组负载。

WENO 的 15 组完整输出通过流式全数组摘要逐项一致；没有为了保存对照而
同时保留两版全域数组。小例同时使用直接逐值比较。显式 OpenMP 的全域
输出也与普通构建的固定来源一致。

| 同轮完整秒数；首轮单列，之后三次中位数 | 固定来源 | 新实现 |
| --- | ---: | ---: |
| 初轮单工作者 | 1.38895 | 1.31843 |
| 初轮四工作者 | 1.19449 | 1.11763 |
| 加入坐标预检后的最终四工作者 | 1.27810 | 1.17087 |

最终组的原始重复范围为旧版 1.16272–1.55974 秒、新版 1.14546–1.20945 秒；
首轮分别为 1.74394 / 1.20274 秒。初轮和最终轮不拼接取比率，也不删除慢样本。
这些结果表明新的组织保持完整工作流的竞争力，没有稳定实质退化；不保证
所有主机、缓存状态或输入都取得同样加速。

### 为什么不能单看 ghost 计时

旧实现用 malloc 后显式遍历清零最终数组，新实现用 NumPy 零初始化，部分
首次触页和写入成本因此推迟到装载/交换。初轮单工作者 ghost 中位数约
0.312 / 0.404 秒，不能据此认定同一个数值 kernel 回退了约三成。

按每次运行先合计分配、内部值复制、scratch 分配和 ghost，旧/新单工作者
中位数约为 0.615 / 0.574 秒，四工作者约为 0.548 / 0.495 秒。这个分段合计
仅解释成本归属；完整流程仍使用实际起止计时，不能把各阶段中位数相加。
本轮不宣称 minmod 或单个交换原语取得算法加速。

## 内存和安装

WENO 准备的旧/新受控上界为 1,537,264,912 / 1,237,804,431 字节。
其中原始内部值和 coarse 不重叠带来 277,880,832 字节的明确活跃存储减少；
其余差异包含几何和预算核算方式，不能全部当作已测 RSS 节省。

B 的最终值约 938 MB，保留的 curl 约 543 MB，后续同时存活决定了整体峰值。
正式比较的进程峰值 RSS 仍约 1.5 GB，并未降低约 278 MB。所有受控数组在
2 GiB 范围内；几何/数组/进程 RSS/文件页缓存分别解释。

最终由独立 sdist 构建普通 wheel，解压到隔离目录，以 `python -I -S` 运行：
15 项检查通过，1 项 OpenMP 检查按普通构建跳过；所有已加载 simesh 模块均
来自安装目录。安装后的 WENO 全域流程再次完成，15 组完整输出匹配固定来源。

产物在 `analysis-core/benchmark-results/n2/package/`：

- `simesh-0.2.0.dev0.tar.gz`：4,223,827 字节，SHA-256：
  `c12f7d30f0315cea9aeac3db4c4f552357ac36d2b786b900df4cb15b26d891a4`。
- `simesh-0.2.0.dev0-cp311-cp311-macosx_11_0_arm64.whl`：6,004,601 字节，
  SHA-256：`4070ef8849af6881a3db19d1c3a072d81ee15974cdb7b3f84806b6212bd020eb`。
- `install-verification.json`、`installed-weno.json` 及 `installed/` 保存独立安装检查。

## 复现与后续

从 `analysis-core/` 执行；比较输出必须使用新目录：

```bash
make build
make test
.venv/bin/python scripts/run_coordinate_comparison.py \
  --donor /Users/astery/.codex/worktrees/d538/simesh \
  --file ../data/weno509_sub_0000.dat \
  --output-dir benchmark-results/new-n2-comparison --workers 4 --repeats 4
SIMESH_OPENMP=1 make build
OMP_NUM_THREADS=1 make test
SIMESH_OPENMP=0 make build
make test
.venv/bin/python setup.py sdist --dist-dir benchmark-results/new-n2-package
.venv/bin/python -m pip wheel benchmark-results/new-n2-package/simesh-0.2.0.dev0.tar.gz \
  --no-deps --no-build-isolation --wheel-dir benchmark-results/new-n2-package
.venv/bin/python scripts/verify_install.py \
  --wheel benchmark-results/new-n2-package/simesh-0.2.0.dev0-cp311-cp311-macosx_11_0_arm64.whl \
  --target benchmark-results/new-n2-package/installed
```

日志为新目录的 `build-n2*.log`、`compare-n2*.log`、`package-n2.log` 和
`verify-n2-weno.log`；原始结果在 `benchmark-results/n2/`。该证据目录约 72 MiB，
加源码/构建增量远低于本轮 2 GiB 暂存额度，机器仍约 30 GiB 磁盘可用。

N2 已完成。下一阶段 N3 接入长追踪/twist、标量/热 LOS、可选几何计划、有界
消费及大输出；区域粗层旧编排、这些科学组件和旧 Dataset/写回/2D 工作流
仍有尚未迁完的部分。E1 重分块不因此变为生产默认，也没有新增真实大输入、
Q、精确足点或 GPU 验收声明。
