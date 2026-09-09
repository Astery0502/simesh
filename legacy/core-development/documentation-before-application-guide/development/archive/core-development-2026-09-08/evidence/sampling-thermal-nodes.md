# 热 LOS 多节点角点复用探索

状态：本轮已关闭，不采用候选。原探索 worktree 已退出；下文路径保留测量时的含义。
原始 JSON/日志归档于主目录 `analysis-core/benchmark-results/sampling-reuse/archive/thermal-nodes/`。

日期：2026-09-08。面向开发协作；已完成候选验证，等待协调任务安排独立评审。

## 范围与固定来源

工作树 `/Users/astery/.codex/worktrees/ea0b/simesh`，共同基线
`a81722188613f45779b350aac212f6359df31ad5`。仅修改独立 `analysis-core/`
热 LOS 与本证据；未修改旧包、主目录、B/curl 候选或 `current.md`。
端点为可审查候选与完整前后比较，随后由协调任务安排独立评审。

本任务环境为 `analysis-core/.venv`，Python 3.11.14、NumPy 2.4.6、Cython 3.3.0。
全部安装、编译、验证和测量通过 `scripts/with_compute_slot.py` 持有
`/Users/astery/science/simesh/analysis-core/benchmark-results/sampling-reuse/compute.lock`。
单组交错测量全程持锁，设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
`VECLIB_MAXIMUM_THREADS=1`；最多四工作者、2 GiB 受控活跃数组。
初始可用磁盘约 31 GiB。只读 WENO 和 N4 安装路径沿用共同约定。

## 候选机制

`thermal_nodes.pxd` 提供热场专用角点缓存；`thermal_leaf` 每次调用创建并初始化，
缓存不会跨叶、slot、场或工作者。仅两个热分量的 16 个 double 和三个实际索引。
每个采样点仍按原表达式在物理 clamp 后计算 q、检查有效范围、floor 与权重。
索引改变就重读角点，因此极短区间、擦边或舍入跨单元不会复用错误角点。
这也意味着没有省去逐点除法和 floor；只减少命中时的数据寻址与角点读取。
没有硬件计数器，不把此机制等同于降低 DRAM 流量。

插值的 x→y→z 算术与顺序保持；Gaussian 坐标、分段、subdivisions、float64、
响应表、log/pow、epsilon、累加与失败计数均未改动。通用 `native.pxd` 未改。
公共热场入口已要求两个分量，本 helper 只用于该私有热路径。

## 验证与测量方法

未修改基线普通构建通过：26 项通过，2 项 OpenMP 专用检查跳过。
本任务匹配编译的基线副本保存在忽略目录
`analysis-core/benchmark-results/thermal-nodes/baseline/`。
该基线与只读 N4 安装的 64² 双视角、1/4 工作者 15 组完整数组哈希均一致。

`thermal_nodes_edges.py` 直接检查编译入口的 300 组组合：正反向、斜向、近轴向，
节点/半节点/物理上界，次正规长度和一个 ulp 的极短段；subdivisions=1/2/4/7；
样本上限 1/17/100000；有限、NaN、负温度、溢出及缺失 slot。
独立进程比较 values/status/samples/entry/exit 全数组字节哈希，失败输出应为 NaN，
样本计数不超限。这里 entry/exit 是显式底层输入；公共入口的 entry/exit 由 WENO
完整图像及既有科学测试验证。

`thermal_nodes_case.py` 与 `compare_n3.py` 相同热请求：原 rho 归一化、显式
0.45–1.65 MK 正弦制造温度、长度单位 1e8 cm，500² 轴向与斜向，subdivisions=4。
这不代表实际快照热力学恢复。分别计密度准备、温度准备、热场构建和两幅积分；
完整成本从打开文件到两幅拥有结果的内存数组交付，不写图像文件。
64² emissivity-first 控制和所有哈希在完整请求计时外，控制不解释为候选收益。

每个版本独立解释器，使用 `-I -S` 并显式插入唯一源码/安装根与本任务依赖路径，
断言 Python 包和热扩展均来自指定根。正式比较每个工作者配置保留首次一对，
再 AB/BA 交错三对；完整数组哈希涵盖数值、entry/exit、状态、样本数和控制结果。

原始构建日志、JSON 和批次脚本均在本任务忽略目录
`analysis-core/benchmark-results/thermal-nodes/`。

## 完整数值与构建结果

- 候选普通构建的既有检查 26 项通过、2 项跳过；随后新增解析检查 4 项通过。
- 基线、候选普通、候选 OpenMP、恢复普通的 300 组边界/失败批次全数组字节哈希一致。
- 普通 64² 双视角的 1/4 工作者，以及全部 16 次 500² 正式运行，15 个完整数组
  的字节哈希一致；正式结果也跨工作者核对一致。所有图像均完成。
- 两幅 500² 图像采样数分别为 209,354,760、237,965,104；64² 控制为 972,404。
- OpenMP 全部 32 项通过；其 64² 双视角在 1/4 工作者下与普通基线完全一致。
  这里是构建/并行一致性检查，不把跨构建模式时间混入正式性能结果。
- 恢复普通构建后 30 项通过、2 项跳过，300 组边界核对通过。末尾构建信息命令
  误引用没有该查询接口的 `preparation`，在所有数值检查通过后报错；保留原日志。
  随后用 `coordinate` 修正查询并补记最终普通构建信息，见 `final-check.log`；
  `thermal_rays`、`native`、`coordinate` 三者均 `enabled=False`。

内核源码最后仅修正模块说明文字；数值实现仍为 `8862be2`。最终说明同步构建未改
优化选项和数值表达式。解析测试最终将斜向方向归一化为单位向量以匹配公开入口，
四项检查再次通过，见 `final-unit-rays.log`。兼容 AMR 沿用 `setup.py` 的串行编译设置。

## 正式效率结果

单位为秒。首次指本批次首次 500² 请求，并非冷磁盘承诺；后续三对按 BA/AB/BA
顺序交错。完整请求交付两幅内存图像，控制和哈希不在其中。

| 工作者/版本 | 首次完整 | 重复 1 | 重复 2 | 重复 3 | 重复中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/基线 | 18.960795 | 19.858410 | 18.887499 | 19.289818 | 19.289818 |
| 1/候选 | 17.849321 | 17.863945 | 17.834606 | 19.875936 | 17.863945 |
| 4/基线 | 7.045379 | 7.127422 | 7.420261 | 7.521022 | 7.420261 |
| 4/候选 | 6.874895 | 6.976584 | 7.155596 | 7.060396 | 7.060396 |

各阶段三次重复中位数如下。各列独立取中位数，因此分项中位数不必相加等于总计。

| 阶段 | 1/基线 | 1/候选 | 4/基线 | 4/候选 |
| --- | ---: | ---: | ---: | ---: |
| 文件读取与密度准备 | 0.819021 | 0.782704 | 0.737049 | 0.757235 |
| 制造温度与准备 | 0.296852 | 0.302936 | 0.297749 | 0.299113 |
| 热场构建、释放原场 | 1.259928 | 1.279817 | 1.284616 | 1.291316 |
| 总准备 | 2.390630 | 2.396610 | 2.282170 | 2.320485 |
| 轴向积分 | 7.745111 | 7.210357 | 2.339512 | 2.097292 |
| 斜向积分 | 8.917367 | 8.508218 | 2.782149 | 2.641861 |
| 双视角积分合计 | 16.925328 | 15.514251 | 5.167895 | 4.758100 |
| 文件到双图像 | 19.289818 | 17.863945 | 7.420261 | 7.060396 |
| 含包加载的完整成本 | 19.376240 | 17.929073 | 7.484919 | 7.123125 |
| 单独的 emissivity-first 控制 | 1.496533 | 1.493045 | 1.324723 | 1.490963 |

双视角积分中位数下降约 8.34%/7.93%，完整请求中位数下降约 7.39%/4.85%
（分别对应 1/4 工作者）。单工作者积分重复范围为基线 16.495459–17.198606、
候选 15.409039–17.474571；四工作者为 4.843954–5.198760、4.739154–4.762271。
单工作者第三对完整结果倒退约 3.0%，保留这对，不认为每次都更快。四工作者
三对完整结果均改善，但控制路径自身也存在波动；未改的准备和控制耗时差异
不解释为候选收益。没有性能计数器证据来区分具体硬件瓶颈。

正式运行峰值 RSS 最大值（十进制 MB）：1 工作者基线/候选 1168.9/1156.5，
4 工作者 1242.3/1177.3。64² 及正式 WENO 测量观测峰值约 1267.6 MB。RSS 波动不解释
为缓存节省了全域内存；新增缓存只含 16 个 double、三个 int64 和一个 bint，
本机结构布局约 160 字节/当前叶调用，至多四个工作者私有副本，没有新增全域数组。

同一热场 `nbytes=625,593,696`。两图保留并核对的五类数组共 20,000,000 字节；
公共结果还有零值 misses 数组和少量几何元数据，未把它们算入该“五类数组”数字。
积分接口自身分配、物理单位转换和返回均计入积分时间；引用交付计入完整时间，
没有磁盘图像编码阶段。本任务日志/基线副本约 43 MiB，环境约 95 MiB，构建缓存
约 12 MiB，远低于新增暂存 2 GiB 上限；没有复制 WENO 文件或保存大图像副本。

`summary.json` 和 `summary.log` 给出各阶段首次、全部重复、中位数及范围；
`formal/*.json` 保留 16 次原始记录、模块路径、构建模式、完整哈希和 RSS。

## 结论与剩余风险

建议保留该候选进入独立评审：保持原数值方法和完整结果的前提下，1/4 工作者
局部及完整成本中位数均改善，额外状态小，变更集中于热采样 helper。
当前没有合并主目录，也不把本结论视为已通过独立评审或与 B/curl 候选组合验收。

收益幅度有限，单工作者有一对倒退；不能宣称固定倍率、任意 subdivisions、
其他 CPU/编译器或真实温度场都受益。热专用 helper 保留了一份两分量插值表达式，
将来若通用插值契约改变，需同步审核。两个路径的边界和失败一致性依赖逐点
实际 q/floor 验证，这些运算不可仅因“同一数学区间”而删除。
制造温度和历史 AIA171 模型的原有科学限制继续适用。

## 复现命令

在本工作树的 `analysis-core/` 下运行，现有原始输出不覆盖；重跑使用新的输出目录。
基线目录是修改内核前普通构建的包副本；跨机器复现应在共同基线单独构建同配置包。
候选实现提交为 `8862be2b696c5a0753ca3c1c274bb74c5b422777`。

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python scripts/with_compute_slot.py \
  --lock /Users/astery/science/simesh/analysis-core/benchmark-results/sampling-reuse/compute.lock \
  -- .venv/bin/python scripts/compare_thermal_nodes.py \
  --baseline benchmark-results/thermal-nodes/baseline --candidate src \
  --size 500 --repeats 4 --output benchmark-results/thermal-nodes/formal-rerun
```

将 `--size` 改为 64、`--repeats` 改为 1 即为先行核对；每次仍检查 workers=1/4。
边界检查同样经 wrapper 运行：

```sh
.venv/bin/python scripts/with_compute_slot.py \
  --lock /Users/astery/science/simesh/analysis-core/benchmark-results/sampling-reuse/compute.lock \
  -- .venv/bin/python -I -S scripts/thermal_nodes_edges.py \
  --source-root src --dependencies .venv/lib/python3.11/site-packages \
  --output benchmark-results/thermal-nodes/edges-rerun.json
```

独立解释器的导入根分别为：

- 基线：`/Users/astery/.codex/worktrees/ea0b/simesh/analysis-core/benchmark-results/thermal-nodes/baseline`。
- 候选：`/Users/astery/.codex/worktrees/ea0b/simesh/analysis-core/src`。
- 依赖：`/Users/astery/.codex/worktrees/ea0b/simesh/analysis-core/.venv/lib/python3.11/site-packages`。

完整构建和恢复的确切命令保存在 `candidate-check.sh`、`openmp-check.sh`，所有脚本
均由共同 wrapper 持锁执行。普通使用 `make build`，OpenMP 使用
`SIMESH_OPENMP=1 make build`，恢复使用 `SIMESH_OPENMP=0 make build`。
