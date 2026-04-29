# Public AMRVAC API Reference

## Scope

This page summarizes the intended user-facing functions exported by
`simesh.amrvac`. It is a compact reference for signatures and return shapes.
For workflow guidance, read `docs/user-guide.md`.

```python
from simesh import amrvac
```

or:

```python
from simesh.amrvac import read_uniform
```

## Public functions

### `open_dataset(path, *, ghost_width=0)`

Open an AMRVAC `.dat` file as a stateful `AMRVACDataSet`.

Use this when you need metadata, repeated operations, mutation, ghost-cell
refresh, or direct dataset methods.

```python
from simesh.amrvac import open_dataset

ds = open_dataset("snapshot.dat", ghost_width=2)
ds.load_data(field_indices=[0, 1])
ds.exchange_ghost_cells()
ds.write_datfile("updated.dat", overwrite=True)
```

### `read_blocks(path, *, field_indices=None, ghost_width=0, include_ghosts=False)`

Read native AMR block data in SFC/Morton order.

Default return shape:

```text
(nleafs, nw, bx, by, bz)
```

With `include_ghosts=True` and `ghost_width > 0`, the block dimensions include
ghost padding.

`field_indices` are zero-based indices into the original file field list.

```python
from simesh.amrvac import read_blocks

blocks = read_blocks("snapshot.dat", field_indices=[0, 1])
```

### `read_uniform(path, *, resolution, bounds=None, field_indices=None, ghost_width=0, interpolation="zero")`

Read AMRVAC data onto a user-facing uniform grid.

Return shape:

```text
(nx, ny, nz, nw)
```

For Cartesian 2D datasets, `resolution` may be `(nx, ny)` or `(nx, ny, 1)`.
The returned z length is one.

```python
from simesh.amrvac import read_uniform

grid = read_uniform(
    "snapshot.dat",
    resolution=(128, 128, 128),
    field_indices=[0],
)
```

Interpolation choices:

| Value | Meaning |
| --- | --- |
| `"zero"` | Piecewise-constant sampling; default low-memory path |
| `"linear"` | Linear sampling through ghost-cell-padded mesh storage |

`interpolation="linear"` requires `ghost_width > 0`.

`field_indices` are zero-based indices into the original file field list. The
returned field axis follows the requested order.

### `write_datfile(path, output_path, *, field_indices=None, ghost_width=0, overwrite=False)`

Write a `.dat` file from an existing `.dat` input. Use `field_indices` to write
a subset of fields.

```python
from simesh.amrvac import write_datfile

write_datfile(
    "snapshot.dat",
    "density_only.dat",
    field_indices=[0],
    overwrite=True,
)
```

### `load_from_uniform(udata, w_names, xmin, xmax, block_nx, **header_updates)`

Create an in-memory `AMRVACDataSet` from user-facing uniform data.

Input `udata` shape:

```text
(nx, ny, nz, nw)
```

For 2D, use `(nx, ny, 1, nw)` and pass `block_nx=(bx, by)`.

```python
import numpy as np
from simesh.amrvac import load_from_uniform

udata = np.zeros((64, 64, 64, 2))
ds = load_from_uniform(
    udata,
    ["rho", "p"],
    np.array([0.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
    np.array([16, 16, 16]),
)
```

### `write_datfile_from_uniform(path, udata, w_names, xmin, xmax, block_nx, overwrite=False, **header_updates)`

Create and write an AMRVAC `.dat` file from user-facing uniform data.

```python
import numpy as np
from simesh.amrvac import write_datfile_from_uniform

udata = np.zeros((64, 64, 64, 2))

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

### `load_uniform_data(path, field_indices=None, return_geometry=True)`

Load level-1 AMRVAC data directly as user-facing uniform data.

When `return_geometry=True`, returns:

```text
(udata, geometry)
```

When `return_geometry=False`, returns only `udata`.

```python
from simesh.amrvac import load_uniform_data

udata, geometry = load_uniform_data("uniform.dat")
```

### `datfile_to_vtk(path, output_path, field_indices=None)`

Convert level-1 AMRVAC data to VTK legacy structured-points output.

```python
from simesh.amrvac import datfile_to_vtk

datfile_to_vtk("uniform.dat", "uniform.vtk", field_indices=[0])
```

## Dataset methods users may need

These methods are available on the object returned by `open_dataset()` or
`load_from_uniform()`.

| Method or attribute | Purpose |
| --- | --- |
| `ds.load_data(field_indices=None)` | Load selected block fields |
| `ds.blocks(include_ghosts=False)` | Return loaded block data |
| `ds.exchange_ghost_cells()` | Refresh ghost cells after mutation |
| `ds.uniform_grid(...)` | Sample loaded data into compute layout |
| `ds.uniform_full(field_indices=None)` | Exact level-1 full-domain placement |
| `ds.write_datfile(path, overwrite=False)` | Write loaded data to `.dat` |
| `ds.wnames` | Field names |
| `ds.domain_nx` | Domain cell counts |
| `ds.block_nx` | Block cell counts |
| `ds.physical_domain` | Domain bounds |

## Layout names

| Name | Shape | Public use |
| --- | --- | --- |
| `udata` | `(nx, ny, nz, nw)` | Preferred user-facing uniform layout |
| `datau` | `(nw, nx, ny, nz)` | Compute-oriented uniform layout |
| `sfc_data` | `(nleafs, nw, bx, by, bz)` | Native block layout |

Helpers:

```python
from simesh.amrvac.layouts import datau_to_udata, udata_to_datau
```
