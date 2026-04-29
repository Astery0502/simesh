# simesh

`simesh` is a Python/Cython toolkit for reading, writing, and sampling
AMRVAC-style adaptive mesh refinement (AMR) data.

The user-facing API lives in `simesh.amrvac`. It gives Python users direct
functions for AMRVAC `.dat` files and NumPy arrays, while the compiled AMR layer
handles forest connectivity, Morton ordering, ghost cells, and uniform-grid
extraction.

## Core user capabilities

- Read AMRVAC `.dat` snapshots into Python arrays
- Extract native AMR block data in SFC/Morton order
- Resample AMR data onto uniform grids for analysis or plotting
- Open mutable dataset objects for metadata, block edits, and write-back
- Create AMRVAC datasets from NumPy uniform-grid arrays
- Write AMRVAC `.dat` files from existing datasets or NumPy arrays
- Export level-1 uniform AMRVAC data to VTK legacy output
- Work with Cartesian 2D data through a singleton-z Python convention

The current public surface is focused on Cartesian 2D and Cartesian 3D AMR data.

## Start here

Start with the native AMR block view when you want to inspect a snapshot without
changing its adaptive structure:

```python
from simesh.amrvac import read_blocks

blocks = read_blocks("snapshot.dat", field_indices=[0, 1])
```

Use a uniform grid only when your next step needs resampled data, such as
plotting or analysis on a fixed Cartesian mesh:

```python
from simesh.amrvac import read_uniform

grid = read_uniform(
    "snapshot.dat",
    resolution=(128, 128, 128),
    field_indices=[0],
)
```

Uniform arrays use the user-facing layout `(nx, ny, nz, nw)`.

## Choose an interface

| Goal | Start with |
| --- | --- |
| Read AMR data on a uniform grid | `read_uniform(...)` |
| Read native AMR block data | `read_blocks(...)` |
| Keep a mutable dataset open | `open_dataset(...)` |
| Build a dataset from NumPy data | `load_from_uniform(...)` |
| Write NumPy data to `.dat` | `write_datfile_from_uniform(...)` |
| Copy or subset an existing `.dat` file | `write_datfile(...)` |
| Load level-1 data as uniform data | `load_uniform_data(...)` |
| Convert level-1 data to VTK | `datfile_to_vtk(...)` |

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
