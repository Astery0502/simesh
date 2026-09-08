# 采样复用探索：共同启动约定

日期：2026-09-08。面向开发协作。用户已授权按“固定 N4 基线、两个独立任务/
worktree、顺序测量、独立评审、最后整合验证”的方式实施本轮探索。

## 本轮端点

只探索两个候选：[B/curl 联合采样](paired-vector.md)与
[热 LOS 多节点复用](thermal-nodes.md)。实现、核心验证、前后完整效率比较、
独立评审和采用/不采用的有据结论均属于授权范围。
当前协调任务管理基线、测量互斥、评审与最终整合；探索任务只交付自身候选。

本轮不进入重分块、读取路径改造、自适应积分、新重构、GPU，也暂不展开
专用 curl、磁力线重分组和 LOS 树遍历改造。不能为扩大加速而换精度、改变
采样/积分工作量、降低接受条件或隐藏准备/输出成本。

## 固定来源与工作区

任务启动消息提供共同基线提交。每个任务从该提交的独立 worktree 工作，
只修改该 worktree 的 `analysis-core/` 和自己对应的证据文档。父目录旧
`src/simesh`、其他任务 worktree 与主目录运行环境都不修改。
新 core、源码来源和 N1–N4 证据见 `next-generation-design.md`、新包 `ASSETS.md`。
`current.md` 的历史部分不覆盖本启动约定。

数值基线为已验收 N4；以下本机产物只读使用：

- WENO：`/Users/astery/science/simesh/data/weno509_sub_0000.dat`。
- N4 安装基线：`/Users/astery/science/simesh/analysis-core/benchmark-results/n4/package/installed`。
- N4 wheel：同目录的 `simesh-0.2.0.dev0-cp311-cp311-macosx_11_0_arm64.whl`。
- 初始化环境可用 Python：`/Users/astery/miniconda3/envs/simesh-dev/bin/python`。
- 后续只用任务目录的 `analysis-core/.venv/bin/python`；建议依赖固定为
  NumPy 2.4.6、Cython 3.3.0，Python 3.11。SciPy 本轮不需要。

先在独立 worktree 验证基线可构建、相关既有检查通过，再修改数值内核。
比较必须用独立解释器，一次只导入一个版本，显式核对模块路径；禁止在一个
解释器中混用两个同名 simesh 的扩展或借用另一个候选的编译产物。

## 计算资源与互斥

本机 8 GiB 内存、8 CPU；一次计算最多四工作者、2 GiB 受控活跃数组。
本轮新增暂存总量控制在 2 GiB 内、至少保留 2 GiB 可用空间。大样本只读共享，
不要复制完整输入、长期保存完整 B/curl 或多份大输出；记录完整结果哈希。

所有编译、测试、剖析和正式测量，均通过同一个本机锁执行：

```bash
.venv/bin/python scripts/with_compute_slot.py \
  --lock /Users/astery/science/simesh/analysis-core/benchmark-results/sampling-reuse/compute.lock \
  -- .venv/bin/python -m pytest tests -q
```

环境初建和任何编译包安装也持锁，可用上述已有 Python 调用 wrapper。
等待锁时可以阅读代码和编写文档，不在锁外另开计算。不要删除或另换锁文件。
基线/候选完整交错比较由同一次锁持有覆盖，避免另一任务插入改变缓存/桌面负载。
设置 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、
`VECLIB_MAXIMUM_THREADS=1`，显式计算 workers 仍可为 1 或 4。

## 数值与效率验收

保持现有 float64、重构 scheme、插值表达式、边界归属、有限值检查、接受前缀、
twist/热响应公式与求和顺序。B/curl 和不同场的存储偏移分别处理；分组/局部
复用不代表可扩大有效覆盖。以完整数组按位一致作为本轮保持原方法的门槛。
若候选无法做到，不自行放宽容差或更换科学目标；报告差异及原因，保留基线。

先用一个明确候选判断收益；只有测量解释了不足，才做针对性调整，通常至多
两种主要实现形式。修复实际缺陷属于正常迭代，不为穷举所有布局开启无界实验。

记录两种成本：相同已准备输入的计算时间，以及相同请求的文件→准备→计算→
结果交付时间。哈希/参考校验放在计时外；不得提前做候选必需工作却不计成本。
保持一个首次结果，再交错至少三对正式重复，报告中位数与波动，不删慢样本。
相同实现的对照与候选使用匹配编译器、优化模式和 worker 配置。

准备、运算、输出和额外内存分别报告。更少的源码访问不直接等于更少 DRAM
流量；没有硬件计数器时，只陈述可验证的调用/算术次数与实际时间。
没有稳定完整收益的候选可以不采用；不能只凭局部循环变快宣布完成优化。

普通构建完成后，按实际改动做 OpenMP 一致性检查，再恢复普通构建。沿用
当前兼容 AMR 串行设置，不改整个项目的编译精度/fast-math 或全局布局。

## 任务交付和评审

候选任务在自己的 worktree 保存仅属于该候选的提交，并提交以下材料：

1. 具体 diff、基线/候选提交、复现命令和模块导入路径。
2. 独立科学/边界组合检查，以及完整数组一致性结果。
3. 原始 JSON/日志、首次和重复数据、局部与完整耗时、峰值内存。
4. 保留或拒绝候选的理由、剩余风险和准确的改动范围。

证据分别写 `docs/analysis-core/evidence/sampling-paired-vector.md` 与
`sampling-thermal-nodes.md`，中文。公共代码/注释英文。不要共同修改主
`current.md` 或把另一任务的候选合入自己的分支。

协调任务随后安排独立评审。评审检查实际 diff、边界与寿命、相同工作量、
测量边界、结果完整性及复杂度是否值得；有问题先修复再复核。采用候选在独立
整合 worktree 合并并验证组合收益，不能直接相加两个单独测得的加速。
