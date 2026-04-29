# Cartesian 2D Guide

## Core convention

AMRVAC Cartesian 2D data is represented as `ndim=2` in file metadata. In Python,
`simesh` keeps uniform and block arrays compatible with the 3D-facing array
conventions by using a singleton z axis.

That means:

| Data | 2D shape |
| --- | --- |
| User-facing uniform data, `udata` | `(nx, ny, 1, nw)` |
| Compute uniform data, `datau` | `(nw, nx, ny, 1)` |
| Block data, `sfc_data` | `(nleafs, nw, bx, by, 1)` |

The z length is always one for Cartesian 2D arrays. The file still records
`ndim=2`.

Read `docs/user-guide.md` first if you are still choosing between
`read_uniform()`, `read_blocks()`, and dataset-object workflows.

## Read 2D data on a uniform grid

For 2D datasets, `read_uniform()` accepts either `(nx, ny)` or `(nx, ny, 1)` as
the requested resolution:

```python
from simesh.amrvac import read_uniform

grid = read_uniform(
    "snapshot_2d.dat",
    resolution=(256, 256),
    field_indices=[0, 1],
)

print(grid.shape)
```

Expected shape:

```text
(256, 256, 1, 2)
```

Linear interpolation uses bilinear sampling for Cartesian 2D data:

```python
smooth_grid = read_uniform(
    "snapshot_2d.dat",
    resolution=(256, 256),
    field_indices=[0],
    ghost_width=1,
    interpolation="linear",
)
```

## Read 2D block data

```python
from simesh.amrvac import read_blocks

blocks = read_blocks("snapshot_2d.dat", field_indices=[0, 1])
print(blocks.shape)
```

The final block axis is the singleton z block dimension:

```text
(nleafs, nw, bx, by, 1)
```

## Create a 2D dataset from NumPy data

For 2D input, keep `udata.shape[2] == 1` and pass a two-entry `block_nx`:

```python
import numpy as np
from simesh.amrvac import load_from_uniform

nx, ny, nw = 128, 128, 2
udata = np.zeros((nx, ny, 1, nw))

ds = load_from_uniform(
    udata,
    ["rho", "p"],
    xmin=np.array([0.0, 0.0]),
    xmax=np.array([1.0, 1.0]),
    block_nx=np.array([16, 16]),
)

print(ds.ndim)
print(ds.domain_nx)
print(ds.block_nx)
```

Expected dataset properties:

```text
2
[128 128]
[16 16]
```

## Write a 2D `.dat` file from NumPy data

```python
import numpy as np
from simesh.amrvac import write_datfile_from_uniform

udata = np.zeros((128, 128, 1, 2))

write_datfile_from_uniform(
    "uniform_2d.dat",
    udata,
    ["rho", "p"],
    xmin=np.array([0.0, 0.0]),
    xmax=np.array([1.0, 1.0]),
    block_nx=np.array([16, 16]),
    overwrite=True,
)
```

The written file uses 2D AMRVAC metadata. The singleton z axis is a Python array
layout convention, not a third physical dimension in the file.

## Exact level-1 uniform data

For level-1 2D datasets, `uniform_full()` returns compute-oriented `datau`
layout:

```python
from simesh.amrvac import open_dataset
from simesh.amrvac.layouts import datau_to_udata

ds = open_dataset("uniform_2d.dat")
ds.load_data()

datau = ds.uniform_full()
udata = datau_to_udata(datau)
```

Shapes:

```text
datau: (nw, nx, ny, 1)
udata: (nx, ny, 1, nw)
```

## Common mistakes

- Do not pass 2D `udata` as `(nx, ny, nw)`. Use `(nx, ny, 1, nw)`.
- Do not pass a 3-entry `block_nx` for 2D construction. Use `(bx, by)`.
- Do not request a 2D resolution with a z value other than one.
- Remember that `read_uniform(..., resolution=(nx, ny))` still returns a
  four-dimensional array.
