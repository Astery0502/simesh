# Python API Map

## Purpose

This document identifies the Python modules that are most relevant for user
workflows and how they relate to the lower-level AMR implementation.
For the independent distribution under `analysis-core/`, use its
[migration guide](../../../MIGRATION.md); native fields and the bundled
stateful compatibility interfaces have separate ownership and execution rules.

For task-oriented user documentation, start with:

- `README.md`
- `docs/README.md`
- `docs/user-guide.md`
- `docs/2d-guide.md`
- `docs/api-reference.md`

## Primary public entrypoints

The clean user-facing API is exported from:

- `src/simesh/amrvac/__init__.py`
- `src/simesh/amrvac/api.py`
- `src/simesh/tools/__init__.py`

The main AMRVAC public functions are:

- `datfile_to_vtk(path, output_path, field_indices=None)`
- `load_from_uniform(udata, w_names, xmin, xmax, block_nx, **kwargs)`
- `load_uniform_data(path, field_indices=None, return_geometry=True)`
- `open_dataset(path, *, ghost_width=0, boundary_conditions=None)`
- `read_blocks(path, *, field_indices=None, ghost_width=0, include_ghosts=False, boundary_conditions=None)`
- `read_uniform(path, *, resolution, bounds=None, field_indices=None, ghost_width=0, interpolation="zero", boundary_conditions=None)`
- `write_datfile(path, output_path, *, field_indices=None, ghost_width=0, overwrite=False, boundary_conditions=None)`
- `write_datfile_from_uniform(path, udata, w_names, xmin, xmax, block_nx, overwrite=False, **header_updates)`

The independent scientific helper namespace `simesh.tools` currently exports:

- `potential_field_green(b3_bottom, xmin, xmax, nz, *, backend="auto", balance_flux=True)`

This helper works directly from NumPy-like arrays and does not require AMRVAC
`.dat` files, AMR block structures, dataset objects, or CT staggered face
fields. It returns a cell-centered magnetic field in component-first layout
`(3, nx, ny, nz)` plus a `PotentialFieldGeometry` dataclass.

The lower-level canonical Python modules live in:

- `src/simesh/amrvac/amrvac_dataset.py`
- `src/simesh/amrvac/derived_fields.py`
- `src/simesh/amrvac/datio.py`

Treat `api.py` as the first user-facing entrypoint, `amrvac_dataset.py` as the
stateful dataset implementation, `derived_fields.py` as the dataset-derived
field registration/materialization layer, and `datio.py` as the canonical
low-level AMRVAC format module.

`read_uniform(..., interpolation="linear")` exposes the canonical
Cython-backed interpolation path: trilinear for 3D data and bilinear for
Cartesian 2D data. It requires ghost-cell storage. Enabled ghost-cell storage
uses `ghost_width >= 2`; `ghost_width=1` is rejected during dataset
construction. Keep ``"zero"`` as the default for compatibility with the
previous piecewise-constant behavior and as the low-memory path when ghost-cell
storage is not wanted. For level-1 data sampled on the native full-domain grid,
`uniform_full()` is the exact block placement path rather than a resampling
path.

Stateful derived-variable workflows live on `AMRVACDataSet`, not on the
file-level wrapper functions. Users register recipes with
`ds.register_derived(...)`, explicitly compute them with
`ds.materialize_fields(...)`, and remove materialized columns with
`ds.drop_derived_fields(...)`. The materialized fields become loaded field
columns in `ds.data` and can be selected by `field_names` through
`ds.blocks(...)`, `ds.uniform_grid(...)`, `ds.uniform_full(...)`, and
`ds.write_datfile(...)`.

First-derivative stencil fields use the same materialization workflow but are
registered declaratively with `ds.register_derivative(name, terms)`, where each
term is `(field_name, axis, coefficient)`. These recipes require
`ghost_width >= 2`, batch requested derivative outputs through the Cython
central-difference backend, and expose only `ghost_width - 1` valid derived
ghost layers; the outermost derived-output ghost layer remains zero.

Ghost-dependent derived recipes currently require dependencies to be original
loaded fields. Materialized derived fields may be selected by name for interior
block, uniform-grid, and write workflows, but they are not exchanged as full
mesh ghost fields.

Keep the selector meanings distinct when changing this surface:

- `field_indices` refer only to original AMRVAC `.dat` header indices.
- `field_names` refer to currently loaded field columns, including
  materialized derived fields.
- Methods that accept both selector forms should reject calls where both are
  supplied.

Physical boundary modes for ghost-cell fills are normalized in
`src/simesh/amrvac/boundary.py` before being passed to the Cython mesh. The
public AMRVAC helpers accept `boundary_conditions` as a single mode string, a
field-name mapping, a nested field/side mapping, or an integer table with shape
`(loaded_field_count, 2 * ndim)`. The supported modes are `cont`, `symm`,
`asymm`, and `noinflow`. The `noinflow` mode requires the corresponding normal
velocity or momentum field to be included in the loaded field set.

Cartesian 2D files are represented with AMRVAC `ndim=2` metadata on disk, but
the Python array API keeps the singleton-z convention. `read_uniform()` accepts
2D resolutions as `(nx, ny)` or `(nx, ny, 1)` and returns `(nx, ny, 1, nw)`.
`load_from_uniform()` and `write_datfile_from_uniform()` use the same
singleton-z convention for user-facing 2D uniform arrays.

OpenMP acceleration is exposed as build/runtime introspection rather than as a
separate resampling API:

- `simesh.utils.openmp_enabled()`
- `simesh.utils.openmp_build_info()`

When the AMR extension is compiled with OpenMP, the Cython uniform-grid kernels
use the same high-level AMRVAC calls and OpenMP thread count is controlled by
standard runtime environment variables such as `OMP_NUM_THREADS`.

## Main objects behind the entrypoints

### Dataset

The canonical dataset object and derived-field mixin are defined in:

- `src/simesh/amrvac/amrvac_dataset.py`
- `src/simesh/amrvac/derived_fields.py`

### Compiled AMR structures

The canonical AMR structures used by `simesh.amrvac` come from:

- `src/simesh/utils/lib/amr/forest.pyx`
- `src/simesh/utils/lib/amr/mesh.pyx`

## Alternate or overlapping API paths

The older Python-first implementation now lives under:

- `src/simesh/legacy/`

This legacy tree is preserved for reference and fallback use, but it is not the
default current API surface.

## Suggested public-API stance

For now, it is safest to think of the public API in three layers:

1. canonical AMRVAC-facing modules under `simesh.amrvac`
2. independent array-based scientific helpers under `simesh.tools`
3. lower-level Cython acceleration modules under `simesh.utils.lib` that should
   usually stay behind the Python interface

## Experimental Native Analysis

The additive `simesh.analysis` namespace provides `open_source`, `open_prepared`,
`prepare`/`prepare_region`, `PreparedPool`, `sample`, `derivative`/`curl`, `trace`/`iter_traces`,
`with_curl`/`CurlPool`, `retrace`, `global_curl`, `Plane`/`sample_plane`,
`integrate_los`/`integrate_los_views`/`orthographic_plane`, `iter_uniform`,
`global_curl_file`, `build_fill_plan`/`FillPlan`, and the explicit `AIA171` thermal
workflow (`thermal_fields`, `emissivity_fields`, `integrate_thermal_los`). These do not redirect
canonical Dataset or file-level APIs. See [usage](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/usage.md) and
[historical acceptance](../../core-development/documentation-before-application-guide/development/archive/core-development-2026-09-08/current-before-freeze.md)
before choosing those parent-package interfaces.

File convenience functions are implemented in `amrvac/analysis_io.py` and lazily
load the explicitly bundled `simesh_rewrite` provider. They support nonperiodic
Cartesian 3D v5 ordinary fields, including validated three-component staggered
tails; CT values are not exposed. Selected file fields become source columns
0..K-1, with original file indices recorded in `FieldSource.original_field_ids`.
`open_prepared` returns a detached canonical bulk-prepared native product by default;
optional `bounds=(lower, upper)` prepares complete intersecting leaves through the
exact-phase regional provider, with original-mesh support and no new cut boundary.
`prepare_region` offers the same selection on an existing source. These prepared
products are the starting point for repeated analysis without dynamic loading;
`open_source` owns a file context for bounded reads/preparation.

Native analysis values use `(slot,x,y,z,component)`, with explicit leaf-to-slot
mapping and per-group remaining halo. Primary preparation supports interiors
only (`halo=0`) or two valid layers. Derived groups are independently allocated.
PreparedPool views are scoped borrows; point/line/image/stream outputs have their
declared owned/sink lifetimes. The old STO provider and raw reader retain their
field-major `(slot,field,x,y,z)` layout at the adapter boundary.

Tracing reports accepted prefixes and optional twist/trajectories; exact boundary
footpoints and Q are not delivered. LOS integrates an explicitly supplied scalar
reconstruction; physical response/EOS remains an input decision. New native
consumers do not silently reinterpret 2D, periodic or non-Cartesian snapshots.

## Array Layouts

The canonical AMRVAC path uses three explicit layout conventions:

- `udata`: user-facing uniform data with shape `(nx, ny, nz, nw)`
- `datau`: compute-oriented uniform data with shape `(nw, nx, ny, nz)`
- `sfc_data`: Morton/SFC block data with shape `(nleafs, nw, bx, by, bz)`
- `bfield`: independent potential-field output with shape `(3, nx, ny, nz)`,
  where components are `b1`, `b2`, and `b3`

For Cartesian 2D data, `nz` and `bz` are always one in Python arrays while
AMRVAC headers and tree block indices remain two-dimensional.

Helpers for converting between `udata` and `datau` live in:

- `src/simesh/amrvac/layouts.py`

## Documentation rule

When adding new user-facing functions:

- document the workflow in `docs/user-guide.md` if users should choose it
- document the function in `docs/api-reference.md` if it is part of the public
  `simesh.amrvac` API
- keep `README.md` as a short entry point with only high-level examples
- document it here if it changes the intended Python API surface
- document implementation details in one of the technical notes under `docs/`

The native namespace combines retained runtime reuse and optional geometric fill
plans. File `value_cache_capacity` defaults to zero; planned preparation uses the
same source-lifetime validation and distinguishes actual from requested value
reads. Thermal LOS requires explicit density/length units and external or
isothermal temperature; its historical AIA171 model does not infer snapshot
thermodynamics. Nonlinear thermal rays use a resident compiled 1--4-worker path
with a Python reference, while scalar LOS/tracing retain their explicit
thread-pool/OpenMP choices. See the usage guide for the distinct reconstructions.
