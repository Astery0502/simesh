# Migration to the current root package

The repository root installs the implementation previously developed under
`analysis-core/`, with the distribution/import name `simesh`. Recreate the root
environment and rebuild compiled extensions after relocating a checkout.
The [Chinese application guide](docs/application-guide.md) is the current user
entry point; [API reference](docs/api-reference.md) gives exact signatures.

## Moved directories and imports

| Previous location or import | Current location or import |
| --- | --- |
| `analysis-core/src`, tests, examples and build files | Root `src`, `tests`, `examples` and build files |
| Previous root implementation and rewrite provider | `legacy/previous/` |
| Earlier Python implementation | `legacy/python-first/` |
| Old development and superseded application documents | `legacy/core-development/documentation-before-application-guide/` |
| `simesh.utils.configurations` | `simesh.tools.configurations` |
| `simesh.utils.openmp_enabled` / `openmp_build_info` | Same names under `simesh.amrvac`, describing its Dataset extension |
| `simesh.utils.lib.amr` | Private `simesh.amrvac._mesh`; prefer public Dataset operations |
| Old `simesh.analysis` consumers | Top-level scientific functions and explicit `simesh.bounded` consumers |

There are no forwarding `simesh.utils` or `simesh.legacy` packages in the current
installation. No runtime import resolves into the repository archive.

## Preserved capabilities and explicit crossings

The eight AMRVAC entrypoints retain their signatures: `open_dataset`,
`read_blocks`, `read_uniform`, `load_from_uniform`, `write_datfile`,
`write_datfile_from_uniform`, `load_uniform_data` and `datfile_to_vtk`.
Mutable Dataset and its private stateful Cython mesh remain separate from
native Source/Fields ownership. `source_from_dataset` copies loaded 3D,
nonperiodic interiors. `write_amrvac` requires full original-mesh coverage and
matching snapshot metadata, and writes ordinary values without halos or CT.

Cartesian 2D and nonperiodic `cont`/`symm`/`asymm`/`noinflow` boundary modes are
retained in the Dataset interface. Native preparation is 3D and currently uses
continuous boundaries. Periodic ghost exchange is rejected. Stateful extensions
remain serial even when the supported native kernels are built with OpenMP.
VTK retains the existing level-1 structured-points and endpoint-coordinate
convention; it is not a polyline or AMR hierarchy exporter.

Original file field indices, restricted Source indices and loaded Dataset names
are distinct selectors. Use actual names when moving a scientific workflow.
Choose numerical schemes, units and physical models explicitly. Old implicit
pool dispatch belongs to the explicit `bounded` APIs. Detailed usage and
save/load behavior are consolidated in the application guide.
