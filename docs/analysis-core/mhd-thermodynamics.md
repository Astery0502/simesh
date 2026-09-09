# 显式 MHD 热力学恢复交付记录

日期：2026-09-09。面向开发协作。

在独立 worktree `/Users/astery/.codex/worktrees/3fd3/simesh` 完成。
初始工作区干净，从 `3899116` 快进到共同基线
`812b827801f4dfad4fb312c9a23ee34753a81bd0`，功能分支为
`codex/mhd-thermodynamics`。没有修改主工作区或父包。

实现入口为 `simesh.physics.mhd`，用户说明与完整示例见
[MHD 热力学接口](../../analysis-core/docs/mhd-thermodynamics.md)。
未修改共享导出、应用入口、几何、诊断、全局 README 或当前方向文档。

## 接口与数值决定

- `IdealMHD` 强制指定 `gamma > 1`、总能量或内能、组成和单位。
  总能量只支持内能、动能与完整磁场能之和；输入为体积能量密度。
  不根据 `e` 的名字、单位标签或文件元数据推测物理定义。
- `MHDUnits` 显式给出密度、动量密度、能量密度的 SI 换算因子，复用
  `MagneticUnits` 的磁场、长度和磁导率。温度复用现有
  `CoronalComposition.temperature`，未重新实现组成换算或通用状态方程。
- `mhd_fields` 返回独立 `Fields`，可选择速度、热压、温度、beta、声速、
  Alfvén 速度、马赫数等量。磁场可来自独立字段组，按叶块编号对齐存储。
  保留守恒量组的选区和叶块顺序，使用输入的共同有效 halo，忽略无效填充。
- 默认遇到非法状态抛出带叶块编号、单元索引和状态位的异常；显式选择
  `invalid="nan"` 时，非法节点所有物理输出为 NaN。状态位与内部/含支撑
  节点计数可检查，不把空间有效支撑误当作物理状态有效性。
- 零磁场不使热力学失效，但 beta 和 Alfvén 马赫数返回 NaN 并标记状态。
  不可表示的诊断单独标记；状态列属于分类数据，不应插值。
- 速度专用输出恰好三列，能交给现有双向追踪；所得曲线是瞬时流线，
  步长仍为坐标长度，不是粒子随时间的轨道。K 单位温度可直接进入
  `thermal_fields`，恢复后的 SI 密度转换为 CGS 时使用 `1e-3`。

## 官方定义核对

查阅了 AMRVAC 当前[方程说明](https://amrvac.org/md_doc_2equations.html)、
[参数说明](https://amrvac.org/md_doc_2par.html)和
[经典 MHD 源码](https://amrvac.org/mod__mhd__phys_8t_source.html)。
经典动量定义为密度乘速度，常用归一化磁导率为一；总能量与内能路径的
压力关系分别对应减去动能、磁能和直接乘以 `gamma-1`。
流体能量、背景场/平衡量分裂、半相对论和相对论、无能量方程及额外能量库
不在本次范围。没有声称覆盖全部 AMRVAC 物理模块。

## 验证

使用本 worktree 的 `analysis-core/.venv`，由主目录已有 Python 3.11
创建；依赖来自可用缓存，在本目录独立构建安装。已确认 Python 包和
原生扩展加载路径均位于本 worktree，没有依赖父包源码或修改主目录环境。

```bash
cd analysis-core
.venv/bin/python -m pytest tests/test_mhd_thermodynamics.py tests/test_standard_diagnostics.py tests/test_derived.py tests/test_science_workflows.py -q
```

结果：39 项通过、1 项普通构建的 OpenMP 用例跳过，其中新增 MHD 用例
24 项。覆盖总能量/内能解析状态、SI/CGS/独立缩放、自定义字段选择和组成、
混合 AMR、选区与存储顺序不一致、不同有效 halo、非法填充、负压/零密度/
非有限输入、数值溢出、零磁场、空覆盖、借用过期及独立所有权。

已直接执行用户文档中的完整混合 AMR 示例：恢复温度约为 800000 K，
6×5 热 LOS 图全部完成，强度约为 123.13348，符合解析柱积分；两种热重建
顺序另有测试覆盖。两个种子的四个速度追踪分支各有 9 个点，符合直线解析解。

核心没有变更，未执行完整历史构建/性能矩阵。剩余限制包括总能量相减的
消减误差，以及先在准备节点恢复再插值所确定的非线性重建顺序；不自动
修复源数据，不生成新 halo，不扩大现有非周期笛卡尔三维边界。
