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

### `open_dataset(path, *, ghost_width=0, boundary_conditions=None)`

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

### `read_blocks(path, *, field_indices=None, ghost_width=0, include_ghosts=False, boundary_conditions=None)`

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

### `read_uniform(path, *, resolution, bounds=None, field_indices=None, ghost_width=0, interpolation="zero", boundary_conditions=None)`

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

### `write_datfile(path, output_path, *, field_indices=None, ghost_width=0, overwrite=False, boundary_conditions=None)`

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

## Boundary conditions for ghost cells

The `boundary_conditions` argument controls physical boundary fills when
`ghost_width > 0`. It is accepted by `open_dataset()`, `read_blocks()`,
`read_uniform()`, and `write_datfile()`.

Supported modes:

| Mode | Meaning |
| --- | --- |
| `"cont"` | Copy the nearest interior cell into the ghost region |
| `"symm"` | Mirror interior cells across the physical boundary |
| `"asymm"` | Mirror interior cells and flip the sign |
| `"noinflow"` | Copy values, but clip inflow on the loaded normal velocity or momentum field |

By default, all loaded fields and sides use `"cont"`.

Accepted forms:

```python
# One mode for every loaded field and physical side.
read_blocks("snapshot.dat", ghost_width=2, include_ghosts=True, boundary_conditions="symm")

# Per-field modes. Field names refer to the loaded AMRVAC field names.
read_blocks(
    "snapshot.dat",
    field_indices=[0, 1, 2],
    ghost_width=2,
    include_ghosts=True,
    boundary_conditions={"rho": "cont", "m1": "noinflow"},
)

# Per-side modes for selected fields.
read_blocks(
    "snapshot.dat",
    field_indices=[0, 1],
    ghost_width=2,
    include_ghosts=True,
    boundary_conditions={"rho": {"xlo": "symm", "xhi": "asymm"}},
)
```

Physical side names are `xlo`, `xhi`, `ylo`, `yhi` for 2D data, with `zlo`
and `zhi` added for 3D data. Integer tables are also accepted with shape
`(loaded_field_count, 2 * ndim)` and mode codes `0=cont`, `1=symm`,
`2=asymm`, `3=noinflow`.

`"noinflow"` requires the corresponding normal velocity or momentum field to
be loaded. Recognized x-direction names include `m1`, `v1`, `u1`, `mom1`,
`rho_v1`, `mx`, `vx`, `ux`, `momx`, and `rho_vx`, with matching y/z variants.

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
| `ds.load_data(field_indices=None, boundary_conditions=None)` | Load selected block fields |
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
