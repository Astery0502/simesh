# simesh

`simesh` is a Python/Cython toolkit for AMRVAC-style adaptive mesh refinement
(AMR) data. It operates on AMRVAC `.dat` snapshots and NumPy arrays, and its
main idea is to keep the native AMR block structure available while also
offering uniform-grid sampling when a downstream tool needs regular Cartesian
arrays.

The main AMRVAC user-facing API lives in `simesh.amrvac`. It gives Python users
direct functions for AMRVAC `.dat` files and NumPy arrays, while the compiled
AMR layer handles forest connectivity, Morton ordering, ghost cells, and
uniform-grid extraction. Independent scientific helpers that do not require
AMRVAC files or AMR meshes live in `simesh.tools`.

## What it works with

- AMRVAC `.dat` snapshots with Cartesian 2D or Cartesian 3D AMR metadata
- Native AMR block arrays in SFC/Morton order
- User-facing NumPy uniform arrays with layout `(nx, ny, nz, nw)`
- Mutable dataset objects that can load metadata, edit blocks, refresh ghost
  cells, and write AMRVAC-compatible output

Uniform grids produced from refined AMR data are sampled or interpolated views.
Use them for plotting, fixed-grid analysis, or array tools that cannot operate
on the adaptive block structure directly.

## Core user capabilities

- Open AMRVAC `.dat` snapshots as mutable dataset objects
- Read native AMR block data in SFC/Morton order
- Register and materialize named derived fields on loaded dataset objects
- Sample AMR data onto uniform grids for analysis or plotting
- Create AMRVAC datasets from NumPy uniform-grid arrays
- Write AMRVAC `.dat` files from existing datasets or NumPy arrays
- Export level-1 uniform AMRVAC data to VTK legacy output
- Work with Cartesian 2D data through a singleton-z Python convention
- Generate independent Cartesian helper fields through `simesh.tools`, such as
  potential-field extrapolations from a bottom `b3` array

## Quick install

Use Python 3.11 or newer. From the repository root:

```bash
pip install .
```

For editable local development:

```bash
pip install -e .
```

Source builds require NumPy and Cython; see `docs/installation.md` for the full
install, build, and test workflow.

## Start here

Use `open_dataset(...)` when you need metadata, repeated operations, block
edits, ghost-cell refresh, or write-back:

```python
from simesh.amrvac import open_dataset

ds = open_dataset("snapshot.dat", ghost_width=2)
print(ds.geometry, ds.domain_nx, ds.block_nx)
print(ds.physical_domain)

ds.load_data(field_indices=[0, 1])
blocks = ds.blocks()
```

Stateful datasets can also register derived fields without treating them as
original file fields:

```python
ds.register_derived(
    "p",
    lambda ctx: ctx.field("e") - 0.5 * ctx.field("rho"),
    dependencies=["rho", "e"],
)
ds.materialize_fields(["p"])
pressure = ds.blocks(field_names=["p"])
```

Use `read_blocks(...)` when you only need the native AMR block payload without
changing its adaptive structure:

```python
from simesh.amrvac import read_blocks

blocks = read_blocks("snapshot.dat", field_indices=[0, 1])
```

Use `read_uniform(...)` only when your next step needs sampled data on a fixed
Cartesian mesh:

```python
from simesh.amrvac import read_uniform

grid = read_uniform(
    "snapshot.dat",
    resolution=(128, 128, 128),
    field_indices=[0],
)
```

This changes the representation from adaptive AMR blocks to a uniform array.
The `interpolation` argument controls how AMR cell values are sampled.

Use `potential_field_green(...)` when you have a uniform bottom-face magnetic
field and want an independent cell-centered Cartesian potential-field box:

```python
import numpy as np
from simesh.tools import potential_field_green

b3_bottom = np.zeros((64, 64))
b3_bottom[24:40, 24:40] = 1.0
bfield, geometry = potential_field_green(
    b3_bottom,
    xmin=[0.0, 0.0, 0.0],
    xmax=[1.0, 1.0, 1.0],
    nz=64,
)

print(bfield.shape)  # (3, 64, 64, 64)
print(geometry.spacing)
```

This helper is independent of AMRVAC file readers, AMR block objects, and
solver-compatible CT staggered fields.

## Choose an interface

| Goal | Start with |
| --- | --- |
| Inspect mesh metadata or keep a mutable dataset open | `open_dataset(...)` |
| Read native AMR block data | `read_blocks(...)` |
| Sample AMR data on a uniform grid | `read_uniform(...)` |
| Build a dataset from NumPy data | `load_from_uniform(...)` |
| Write NumPy data to `.dat` | `write_datfile_from_uniform(...)` |
| Copy or subset an existing `.dat` file | `write_datfile(...)` |
| Load level-1 data exactly as uniform data | `load_uniform_data(...)` |
| Convert level-1 data to VTK | `datfile_to_vtk(...)` |
| Build an independent potential field from bottom `b3` | `simesh.tools.potential_field_green(...)` |

## Documentation

Start with the page that matches your role:

- `docs/README.md`: documentation index by audience and task
- `docs/installation.md`: install, editable builds, tests, and OpenMP
- `docs/user-guide.md`: user workflows and examples
- `docs/2d-guide.md`: Cartesian 2D singleton-z behavior
- `docs/api-reference.md`: public `simesh.amrvac` API reference
- `docs/python-api-map.md`: public API map for maintainers
- `docs/architecture.md`: project layout and implementation layers
- `docs/amrvac-dat-format.md`: AMRVAC `.dat` format notes
- `docs/amr-forest-mesh.md`: AMR forest, mesh, and ghost-cell notes
- `docs/cython-build.md`: Cython extension build details
- `docs/performance-benchmarks.md`: benchmark workflow

## Limits of the current user surface

`simesh` currently targets Cartesian 2D and Cartesian 3D AMR meshes. Ghost-cell
fills support continuous, symmetric, antisymmetric, and no-inflow physical
boundary modes on those Cartesian meshes. Broader geometry support should be
treated as outside the current stable user workflow.
