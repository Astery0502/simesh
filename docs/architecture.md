# Architecture

The repository root builds the native AMR analysis package previously developed
in `analysis-core/`. Its import and distribution name remains `simesh`.
See the [Chinese application guide](application-guide.md) for scientific
workflows and [migration guide](../MIGRATION.md) for moved interfaces.

## Native data and calculation

`mesh.py` owns shared integer topology and physical geometry. `io/` supplies
immutable file/array Sources and detached snapshot metadata. `preparation/`
reads explicit support and publishes independent `Fields`; `fields.py` keeps
storage halo separate from valid halo. Scientific consumers operate on these
completed fields. `bounded.py` explicitly coordinates limited-capacity input
preparation for its supported consumers.

`operators/`, `tracing.py`, `projection.py`, `connectivity.py`, `physics/` and
`reductions.py` expose scientific operations. Their hot loops use Cython under
`_kernels/`. `_amr/` supplies Python validation and bindings for the retained
AMR routines under `_kernels/primitives/`. These routines are current runtime
code even when their numerical algorithms originated in preceding versions.

`geometry.py`, `applications.py` and `line_profiles.py` associate results with
identified points, rays and curves. `results_io.py` and `result_shards.py`
provide explicit numerical product delivery. No consumer silently expands
coverage or changes the preparation scheme to meet an execution constraint.

## Stateful AMRVAC and array tools

`amrvac/` retains mutable Dataset, ordinary file read/write, 2D singleton-z and
legacy structured-points VTK workflows. Its `_mesh/` folder contains only the
stateful Cython mesh, forest, Morton implementation and support headers needed
by these workflows. It replaces the old `utils/lib/amr/` hierarchy.

Native preparation, sampling, tracing and LOS do not call that mutable AMRMesh.
`io/products.py` provides two explicit boundaries: copying loaded Dataset
interiors into a Source and serializing complete native Fields through the
ordinary AMRVAC writer. File export does not make the compatibility mesh an
owner of native data.

`tools/potential_field.py` and `tools/configurations.py` work on arrays without
an AMR Dataset. OpenMP introspection for the stateful mesh is exposed through
`simesh.amrvac.openmp_build_info`; native kernel build settings are separate.

## Archives and builds

`legacy/previous/` holds the preceding main implementation;
`legacy/python-first/` holds its earlier Python reference implementation.
Historical core-development runners/artifacts are under
`legacy/core-development/`. These directories are excluded from package
discovery and default pytest collection. There is no active `simesh.utils`,
`simesh.legacy` or `simesh_rewrite` package.

`setup.py` builds the current `_kernels/` and `amrvac/_mesh/` extensions only.
See [Cython build notes](cython-build.md) for compiler groups and wheel checks.
