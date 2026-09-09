# 新版提升为根目录核心

日期：2026-09-09。面向开发协作。

按用户决定，在当前 `codex/analysis-core-p0-p4` 分支将新版提升为默认项目，
目录清晰优先，允许调整旧导入路径。迁移包含工作区中原有的未提交修改与新增文件。
本轮不修改物理算法、数值格式或结果文件协议。

## 当前布局

- 原 `analysis-core/` 的源码、测试、示例和构建配置已提升到仓库根目录。
- 新版用户文档位于 `docs/`；开发协作与历史证据位于 `docs/development/`。
- `legacy/previous/` 保存上一代主包、`rewrite`、旧测试、构建工具及配套资料。
- `legacy/python-first/` 单独保存更早的 Python 实现和包形成前的源码。
  其中 `src/simesh/legacy/` 保留原始源码命名，不是当前可导入的命名空间。
- `legacy/core-development/` 保存历史核心比较脚本、日志和本机环境等产物。

历史源码用于参考；旧测试和脚本中的原始路径与导入假设不保证在归档位置直接可用。
需要重现实验时，应使用对应 Git 版本和独立环境。当前构建不读取这些归档。

## 当前计算代码与导入变化

原生科学计算继续使用 `_kernels/` 与 `_kernels/primitives/`。经依赖检查，
可变 Dataset、二维读写及均匀网格兼容功能仍需要旧式网格实现，故将必要代码收拢到
`src/simesh/amrvac/_mesh/`，不把运行依赖放进历史归档。

| 原路径 | 当前路径 |
| --- | --- |
| `simesh.utils.lib.amr` | 私有 `simesh.amrvac._mesh`，支撑头文件与扩展同目录 |
| `simesh.utils.configurations` | `simesh.tools.configurations` |
| `simesh.utils.openmp_enabled` / `openmp_build_info` | `simesh.amrvac` 下的同名入口 |

不保留 `simesh.utils` 转发层。原生科学接口与八个 AMRVAC 主入口保持现有调用方式。
根目录重新建立开发环境；`.[dev]` 提供本地直接构建所需的 Cython、setuptools、wheel
和 pytest，`.[test]` 仍只提供测试依赖。

## 验证结果

- 根目录可编辑安装成功；完整测试 **290 项通过、2 项 OpenMP 用例跳过**。
- 构建源码包，再从该源码包构建 wheel；确认不包含历史包或被移除的命名空间。
- 提取 wheel 后以隔离解释器运行测试，同样 **290 项通过、2 项跳过**。
  安装检查也约束测试子进程的导入路径，避免读取本地可编辑源码。
- 两个代表性应用示例成功运行，物理汇总和结果重新加载通过；迁移后的解析场工具及
  兼容扩展状态入口检查通过。
- 对照迁移前工作区副本，迁移输入文件无遗漏；410 个归档 Python/Cython 源码和头文件
  内容未变。历史文档的相对链接按新位置调整，未改写历史数值结论。

这些检查针对结构迁移、安装独立性和已有行为，不增加新的物理或性能验收承诺。
当前用户入口见[项目说明](../../../../README.md)、[迁移说明](../../../../MIGRATION.md)及
[物理能力与输出指南](../capabilities-and-outputs.md)。
