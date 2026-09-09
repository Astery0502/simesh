# 探索任务：B 与 curl 联合采样

先读[共同约定](sampling-reuse.md)。本任务只研究 twist 中同一位置对 B 与
curl(B) 采样时的几何/权重复用，不更换 curl 算法、RK 方法、追踪工作组织或
场存储。普通 B-only 追踪继续保留原路径。

## 源码切入点与假设

当前 `_kernels/native.pyx` 的 `_advance_line_range` 在同一 RK 位置分别调用
两次 `interpolate`；`_kernels/native.pxd` 计算局部坐标、floor、索引与权重。
B 与 curl 共享 Mesh 和物理点，但可能有不同 slot、storage_halo 与有效范围。

候选假设：在一次联合采样中只计算一次兼容的单元位置和权重，再分别读取
三个 B 和三个 curl 分量，可减少重复计算且保持浮点结果。不能把两个字段
拼成全域六分量副本，也不能用新的解析 curl 替代已准备 curl。

优先将新 helper 放在任务自有头文件，尽量不修改通用 `native.pxd`。若必须
调整共享接口，在结果中明确列出，协调任务负责与热候选的整合。

## 核心检查

覆盖存储 halo 不同、非打包 slot、粗细/物理边界位置、无效采样和数值极值，
检查 finite/null/termination 语义。用已有螺旋场验证 twist 和接受前缀；
兼顾直接 Fields 与小容量 CurlPool 恢复路径，不能让失败的联合采样修改状态。
复用已有核心检查，新增项仅回答新 helper 的具体正确性问题。

## 有用的前后比较

以已准备 B 与 curl 为相同输入，对照固定 N4 的 twist 追踪。先核对小型路径，
然后使用 WENO：256² XY 种子、归一化 z=.05，XY 分布沿用 `compare_n3.py`，
步长为最小间距的 .25，最多 1000 步，开启 twist、仅保留摘要；另用小组选中
种子核对完整轨迹。该负载用于实际大量联合采样，不能只测原 N3 的 64 种子诊断。

记录相同准备场下的计算成本，以及文件→coordinate-phase 准备→全域 curl→
同一 twist 请求的完整成本。独立进程避免两版大 B/curl 同时驻留；输出哈希
覆盖 seed ID、位置、长度、步数、状态、样本数、twist 和小组轨迹。
普通无 twist 追踪作为必要的回归控制。重点比较 workers=1/4；不展开调度矩阵。

最终交付按共同约定，证据写 `evidence/sampling-paired-vector.md`。如果完整
收益不明显，也提交准确结论和可审查补丁，不以扩大任务范围获取加速。
