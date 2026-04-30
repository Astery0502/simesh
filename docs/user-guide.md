# User Guide

## Scope

This guide is for users who want to load, inspect, sample, modify, or write
AMRVAC-style data from Python. `simesh` operates on two main inputs:

- AMRVAC `.dat` snapshots, including Cartesian 2D and Cartesian 3D AMR data
- NumPy arrays that represent uniform Cartesian data

The central distinction is native AMR data versus derived uniform-grid data.
Native AMR block arrays preserve the adaptive block structure and SFC/Morton
ordering from the file. Uniform arrays are regular Cartesian arrays for
plotting, fixed-grid analysis, or downstream tools that do not understand AMR
blocks. When refined AMR data is read through `read_uniform(...)`, the result is
a sampled or interpolated view, not the exact native block representation.

The main public namespace is:

```python
import simesh.amrvac as amrvac
```

Most workflows should use the functions exported from `simesh.amrvac` instead
of importing lower-level modules directly.

If you have not installed the package yet, see `docs/installation.md`.

## Choose a workflow

| Workflow | Start with | Returns or creates |
| --- | --- | --- |
| Inspect metadata, mutate blocks, or write back | `open_dataset(...)` | `AMRVACDataSet` |
| Read native AMR block values | `read_blocks(...)` | `sfc_data`, `(nleafs, nw, bx, by, bz)` |
| Sample AMR data onto a uniform grid | `read_uniform(...)` | `udata` array, `(nx, ny, nz, nw)` |
| Build a dataset from NumPy data | `load_from_uniform(...)` | `AMRVACDataSet` |
| Write NumPy data to AMRVAC `.dat` | `write_datfile_from_uniform(...)` | write summary dict |
| Load level-1 data exactly as uniform data | `load_uniform_data(...)` | `udata`, optionally geometry |
| Export level-1 data to VTK | `datfile_to_vtk(...)` | `.vtk` file |

## Work with a dataset object

Use `open_dataset()` when you need metadata, repeated operations, mutation, or
manual ghost-cell refresh:

```python
from simesh.amrvac import open_dataset

ds = open_dataset("snapshot.dat", ghost_width=2)

print(ds.wnames)
print(ds.geometry)
print(ds.domain_nx)
print(ds.block_nx)
print(ds.physical_domain)
print(ds.periodic)

ds.load_data(field_indices=[0, 1])
blocks = ds.blocks()
blocks[:, 0] *= 1.01

ds.exchange_ghost_cells()
ds.write_datfile("updated.dat", overwrite=True)
```

You can pass `boundary_conditions` to `open_dataset()` or override them on
`ds.load_data(...)` before refreshing ghost cells.

### Inspect mesh metadata

Opening a dataset reads the file header and AMR tree metadata before loading
the full block payload. Use these attributes to check the mesh before deciding
whether to inspect blocks, sample a uniform grid, or write a modified file:

| Attribute | Meaning |
| --- | --- |
| `ds.wnames` | Field names stored in the file |
| `ds.geometry` | AMRVAC geometry label, such as `Cartesian_3D` |
| `ds.domain_nx` | Full-domain cell counts in active dimensions |
| `ds.block_nx` | Cell counts per AMR block |
| `ds.physical_domain` | Domain bounds as `(xmin, xmax)` |
| `ds.periodic` | Periodicity flags by active dimension |
| `ds.nleafs` | Number of leaf blocks in the AMR forest |
| `ds.levmax` | Maximum AMR refinement level in the file |

For exact header values, inspect `ds.metadata`, which keeps the parsed AMRVAC
header dictionary:

```python
print(ds.metadata["time"])
print(ds.metadata["it"])
print(ds.metadata["xmin"], ds.metadata["xmax"])
```

Use `ds.uniform_grid(...)` when you intentionally need the lower-level
compute-oriented sampled uniform-grid layout:

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

Prefer the top-level `read_uniform()` helper when you want the user-facing
`udata` layout directly. Convert lower-level `datau` to `udata` with:

```python
from simesh.amrvac.layouts import datau_to_udata

udata = datau_to_udata(datau)
```

## Read block data

Use `read_blocks()` when you want the native AMR block payload without opening a
mutable dataset object:

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

## Set physical boundary conditions

When you request ghost-cell storage with `ghost_width > 0`, physical domain
boundaries default to continuous fills. Pass `boundary_conditions` when you
need a different physical boundary mode:

```python
ghosted = read_blocks(
    "snapshot.dat",
    field_indices=[0, 1, 2],
    ghost_width=2,
    include_ghosts=True,
    boundary_conditions={
        "rho": "symm",
        "m1": {"xlo": "noinflow", "xhi": "noinflow"},
    },
)
```

Supported modes are:

| Mode | Use when |
| --- | --- |
| `"cont"` | Ghost cells should copy the nearest interior value |
| `"symm"` | Ghost cells should mirror values across the physical boundary |
| `"asymm"` | Ghost cells should mirror values and flip the sign |
| `"noinflow"` | The normal velocity or momentum component should be clipped to prevent inflow |

You can pass one mode for every loaded field, a mapping from field name to
mode, or a nested mapping from field name to side name. Side names are `xlo`,
`xhi`, `ylo`, `yhi` in 2D and add `zlo`, `zhi` in 3D.

`"noinflow"` requires the matching normal velocity or momentum field to be
loaded. Common names such as `m1`, `v1`, `mx`, and `vx` are recognized for the
x direction, with corresponding y and z names.

## Sample a uniform grid

Use `read_uniform()` when your next step needs data on a regular Cartesian grid,
such as plotting or fixed-grid analysis:

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

This is a representation change for refined AMR data: the adaptive blocks are
sampled onto the requested grid. `field_indices` are zero-based indices into the
file's original field list. When you pass a list, the output field axis follows
that list order.

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
| `interpolation="zero"` | You want the default low-memory, piecewise-constant sampling path |
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

Use `load_uniform_data()` instead when the file is known to be level-1 and you
want exact full-domain uniform placement rather than AMR resampling.

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

- Prefer `open_dataset()` when you will mutate data or call multiple dataset
  methods.
- Prefer `read_blocks()` when AMR block structure matters.
- Prefer `read_uniform()` for plotting, analysis, and downstream array tools
  that require regular grids.
- Prefer `load_from_uniform()` when constructing an in-memory dataset before
  additional operations.
- Prefer `write_datfile_from_uniform()` for direct NumPy-to-`.dat` output.

## Current limits

- The stable user workflows target Cartesian 2D and Cartesian 3D data.
- `load_uniform_data()` and `datfile_to_vtk()` are level-1 uniform-data
  conveniences.
- Linear interpolation requires `ghost_width > 0`.
