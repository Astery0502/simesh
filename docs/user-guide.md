# User Guide

## Scope

This guide is for users who want to load, sample, modify, or write AMRVAC-style
data from Python. The main public namespace is:

```python
import simesh.amrvac as amrvac
```

Most workflows should use the functions exported from `simesh.amrvac` instead
of importing lower-level modules directly.

If you have not installed the package yet, see `docs/installation.md`.

## Choose a workflow

| Workflow | Start with | Returns or creates |
| --- | --- | --- |
| Sample AMR data onto a uniform grid | `read_uniform(...)` | `udata` array, `(nx, ny, nz, nw)` |
| Inspect native AMR block values | `read_blocks(...)` | `sfc_data`, `(nleafs, nw, bx, by, bz)` |
| Mutate block data and write it back | `open_dataset(...)` | `AMRVACDataSet` |
| Build a dataset from NumPy data | `load_from_uniform(...)` | `AMRVACDataSet` |
| Write NumPy data to AMRVAC `.dat` | `write_datfile_from_uniform(...)` | write summary dict |
| Load level-1 data exactly as uniform data | `load_uniform_data(...)` | `udata`, optionally geometry |
| Export level-1 data to VTK | `datfile_to_vtk(...)` | `.vtk` file |

## Read a uniform grid

Use `read_uniform()` when you want analysis-ready data on a regular grid:

```python
from simesh.amrvac import read_uniform

grid = read_uniform(
    "snapshot.dat",
    resolution=(128, 128, 128),
    field_indices=[0, 1],
)
```

The returned array uses the user-facing `udata` layout:

```text
(nx, ny, nz, nw)
```

`field_indices` are zero-based indices into the file's original field list.
When you pass a list, the output field axis follows that list order.

Use `bounds=(xmin, xmax)` to sample a subdomain:

```python
grid = read_uniform(
    "snapshot.dat",
    resolution=(64, 64, 64),
    bounds=([0.25, 0.25, 0.25], [0.75, 0.75, 0.75]),
    field_indices=[0],
)
```

Choose interpolation by intent:

| Mode | Use when |
| --- | --- |
| `interpolation="zero"` | You want the default low-memory, piecewise-constant path |
| `interpolation="linear"` | You want smoother sampling and can open ghost-cell storage |

```python
smooth_grid = read_uniform(
    "snapshot.dat",
    resolution=(128, 128, 128),
    field_indices=[0],
    ghost_width=1,
    interpolation="linear",
)
```

## Read block data

Use `read_blocks()` when you want the native AMR block payload:

```python
from simesh.amrvac import read_blocks

blocks = read_blocks("snapshot.dat", field_indices=[0, 1])
```

The returned block array uses SFC/Morton block order:

```text
(nleafs, nw, bx, by, bz)
```

To include ghost-cell padded storage, request ghost cells explicitly:

```python
ghosted = read_blocks(
    "snapshot.dat",
    field_indices=[0, 1],
    ghost_width=2,
    include_ghosts=True,
)
```

## Work with a dataset object

Use `open_dataset()` when you need metadata, repeated operations, mutation, or
manual ghost-cell refresh:

```python
from simesh.amrvac import open_dataset

ds = open_dataset("snapshot.dat", ghost_width=2)
ds.load_data(field_indices=[0, 1])

print(ds.wnames)
print(ds.domain_nx)
print(ds.physical_domain)

blocks = ds.blocks()
blocks[:, 0] *= 1.01

ds.exchange_ghost_cells()
ds.write_datfile("updated.dat", overwrite=True)
```

Use `ds.uniform_grid(...)` when you need the lower-level compute-oriented
layout:

```python
datau = ds.uniform_grid(
    (128, 128, 128),
    field_indices=[0],
    interpolation="zero",
)
```

`ds.uniform_grid(...)` returns `datau` layout:

```text
(nw, nx, ny, nz)
```

Prefer the top-level `read_uniform()` helper when you want `udata` directly.
Convert it to user-facing layout with:

```python
from simesh.amrvac.layouts import datau_to_udata

udata = datau_to_udata(datau)
```

## Build or write from uniform NumPy data

Use `load_from_uniform()` when you want an in-memory dataset:

```python
import numpy as np
from simesh.amrvac import load_from_uniform

udata = np.zeros((64, 64, 64, 2))
ds = load_from_uniform(
    udata,
    ["rho", "p"],
    xmin=np.array([0.0, 0.0, 0.0]),
    xmax=np.array([1.0, 1.0, 1.0]),
    block_nx=np.array([16, 16, 16]),
)
```

Use `write_datfile_from_uniform()` when you only need to write a `.dat` file:

```python
from simesh.amrvac import write_datfile_from_uniform

write_datfile_from_uniform(
    "uniform.dat",
    udata,
    ["rho", "p"],
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
    np.array([16, 16, 16]),
    overwrite=True,
)
```

`block_nx` must divide the domain cell count exactly in every active dimension.

## Load level-1 uniform data directly

For level-1 datasets, `load_uniform_data()` provides a direct uniform-grid
loader. Use `read_uniform()` for refined AMR data or for explicit resampling.

```python
from simesh.amrvac import load_uniform_data

udata, geometry = load_uniform_data("uniform.dat")
```

The geometry dictionary includes domain bounds, cell counts, field names,
geometry, periodicity, dimension count, time, and iteration.

## Export to VTK

For level-1 AMRVAC data:

```python
from simesh.amrvac import datfile_to_vtk

datfile_to_vtk("uniform.dat", "uniform.vtk")
```

## Array layout summary

| Name | Shape | Used for |
| --- | --- | --- |
| `udata` | `(nx, ny, nz, nw)` | User-facing uniform arrays |
| `datau` | `(nw, nx, ny, nz)` | Compute-oriented uniform arrays |
| `sfc_data` | `(nleafs, nw, bx, by, bz)` | AMR block data in SFC/Morton order |

Conversion helpers:

```python
from simesh.amrvac.layouts import datau_to_udata, udata_to_datau
```

## Common choices

- Prefer `read_uniform()` for plotting, analysis, and downstream array tools.
- Prefer `read_blocks()` when AMR block structure matters.
- Prefer `open_dataset()` when you will mutate data or call multiple dataset
  methods.
- Prefer `load_from_uniform()` when constructing an in-memory dataset before
  additional operations.
- Prefer `write_datfile_from_uniform()` for direct NumPy-to-`.dat` output.

## Current limits

- The stable user workflows target Cartesian 2D and Cartesian 3D data.
- `load_uniform_data()` and `datfile_to_vtk()` are level-1 uniform-data
  conveniences.
- Linear interpolation requires `ghost_width > 0`.
