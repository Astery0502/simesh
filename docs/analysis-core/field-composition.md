# 准备后字段组合

更新：2026-09-09。面向开发协作。

本轮在独立工作区 `75cd/simesh`、分支 `codex/field-composition` 完成，
从干净状态快进到共同基线 `4f6fc304a866db26dd3c119c53e65549a863745f`。
没有修改父包、共享公共导出、应用入口、结果文件模块或全局文档。

## 已实现接口

- `simesh.field_ops.select_fields`：名称、零起始索引或混合选择，保持显式顺序；
  支持完整输出名称序列，保留单位和解释。拒绝空选择、重复分量和歧义名称。
- `simesh.field_ops.merge_fields`：按输入顺序合并，校验相同 Mesh 对象及叶覆盖；
  对每个输入独立查找叶目录，支持不同物理槽序与选择序。
- `simesh.operators.derived.derive_many`：每叶一次回调返回多个命名分量；
  定义映射指定名称及单位，或使用显式 `FieldDefinition` 序列。
  原有 `derive` 接受标量或块数组的调用方式保持不变。

三个入口均返回紧凑、独立拥有的只读数组，没有共享借用模式。
选择保留输入有效 halo，合并与配方保留共同有效 halo；输出不携带无效 padding。
内存准入计入输入真正的 NumPy 底层数组、输出、叶目录与配方结果块。
多输出配方额外预留所有结果块及一个转换块，因此单输出包装现在预留两个块，
接近原先最小预算的调用可能需要增加一个块的额度。

所有组合产生新的 `value_identity` 并清除 `derivation`，包括同序复制和仅改名。
选择保留来源及 scheme，合并保存有序来源 token 和组合 scheme。
没有通过保留父 Fields 或父值数组来证明来源。采用保守身份规则：
组合后需要针对新的 B 计算 curl，旧证明不会穿过重排、合并或复制。

配方仍然是逐点约定，不能用空间移位、梯度或空间归约替代显式空间算子。
Python 回调自身的任意分配无法由接口限制。标签不做物理换算，
调用者负责输入的时刻、单位、数值重构及物理意义兼容。

## 验证

仅使用本工作区 `analysis-core/.venv`，首次可编辑安装从本地源码生成本地扩展。
已核对 Python 包与原生扩展的实际加载路径均位于本工作区。
本轮没有修改 Cython，因此安装完成后没有强制重建核心。

运行以下针对性检查，共 **81 项通过、1 项跳过**；跳过项是普通构建不支持的
可选 OpenMP 路径。新增组合用例为 27 项。

```bash
analysis-core/.venv/bin/python -m pytest \
  analysis-core/tests/test_field_composition.py \
  analysis-core/tests/test_derived.py \
  analysis-core/tests/test_standard_diagnostics.py \
  analysis-core/tests/test_applications.py \
  analysis-core/tests/test_connectivity.py \
  analysis-core/tests/test_native_workflow.py \
  analysis-core/tests/test_science_workflows.py -q
```

覆盖非连续及混合选择、重命名和单位、物理槽与选择序独立变化、
不同有效 halo、无效 padding、冲突和错误覆盖、借用到期及独立结果、
父大数组内存计数、配方执行前拒绝、回调内部关闭借用等边界。
代表流程将热力学标量与 B 分量合并，再提取 B 接 curl 和 QSL，
筛选后调用默认双向应用追踪；另验证旧 curl 及重排 curl 的来源拒绝。
未运行完整核心基准矩阵或真实强 QSL/大数据验收。

## 整合建议

接口与完整示例见[字段组合文档](../../analysis-core/docs/field-composition.md)。
后续整合可以从公共 `__init__.py` 导出三个入口，并在包 README 及当前方向页
增加文档链接，同时把原有单输出配方内存说明更新为两个结果块。
本轮遵守并行编辑边界，未修改这些共享文件。

MHD 恢复入口无需改变；其返回的普通 Fields 可按现有语义参与组合。
组合产生的 Fields 可直接传给既有消费者，无需修改结果文件 schema。
若将来要把组合配方或 Fields 自身持久化，应另行设计来源及定义的保存契约；
本轮不把运行期身份 token 扩展为跨进程或跨文件认证。
