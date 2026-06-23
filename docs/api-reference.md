# Public API Reference

## Scope

This page summarizes the intended user-facing functions exported by
`simesh.amrvac` and the independent helpers exported by `simesh.tools`. It is a
compact reference for signatures and return shapes. For workflow guidance, read
`docs/user-guide.md`.

```python
from simesh import amrvac
```

or:

```python
from simesh.amrvac import read_uniform
```

Independent tools are imported from:

```python
from simesh.tools import potential_field_green
```

## AMRVAC functions

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

Dataset objects also support explicit derived fields. Derived names are
in-memory loaded field names, not original `.dat` field indices:

```python
ds.load_data(field_indices=[0, 2])
ds.register_derived(
    "p",
    lambda ctx: ctx.field("e") - 0.5 * ctx.field("rho"),
    dependencies=["rho", "e"],
)
ds.materialize_fields(["p"])
blocks = ds.blocks(field_names=["p"])
```

First-derivative stencil fields use declarative terms and materialize through
the same method:

```python
ds = open_dataset("snapshot.dat", ghost_width=2)
ds.load_data(field_indices=[0, 1, 2])
ds.register_derivative("j1", [("b3", "y", +1.0), ("b2", "z", -1.0)])
ds.materialize_fields(["j1"])
```

Derivative fields require `ghost_width >= 2`. Their materialized padded output
has valid interior values and `ghost_width - 1` valid ghost layers; the
outermost derived-output ghost layer remains zero. Ghost-dependent derived
recipes currently require dependencies to be original loaded fields because
materialized derived fields do not have a full ghost-cell exchange contract.

### `read_blocks(path, *, field_indices=None, ghost_width=0, include_ghosts=False, boundary_conditions=None)`

Read native AMR block data in SFC/Morton order.

Default return shape:

```text
(nleafs, nw, bx, by, bz)
```

With `include_ghosts=True` and `ghost_width >= 2`, the block dimensions include
ghost padding. `ghost_width=1` is rejected.

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

`interpolation="linear"` requires ghost-cell storage. Enabled ghost-cell
storage uses `ghost_width >= 2`; `ghost_width=1` is rejected during dataset
construction.

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
`ghost_width >= 2`. It is accepted by `open_dataset()`, `read_blocks()`,
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
| `ds.register_derived(name, func, dependencies, requires_ghosts=False)` | Register a derived-field recipe without computing it |
| `ds.register_derivative(name, terms)` | Register one scalar first-derivative stencil field |
| `ds.materialize_fields(names)` | Compute registered derived fields into the loaded field axis |
| `ds.drop_derived_fields(names)` | Remove materialized derived fields from loaded data |
| `ds.blocks(include_ghosts=False, field_indices=None, field_names=None)` | Return loaded block data |
| `ds.exchange_ghost_cells()` | Refresh ghost cells after mutation |
| `ds.uniform_grid(..., field_indices=None, field_names=None)` | Sample loaded data into compute layout |
| `ds.uniform_full(field_indices=None, field_names=None)` | Exact level-1 full-domain placement |
| `ds.write_datfile(path, overwrite=False, field_names=None)` | Write loaded data to `.dat` |
| `ds.wnames` | Field names |
| `ds.loaded_field_names` | Names matching the loaded columns in `ds.data` |
| `ds.derived_field_names` | Materialized derived field names |
| `ds.domain_nx` | Domain cell counts |
| `ds.block_nx` | Block cell counts |
| `ds.physical_domain` | Domain bounds |

`field_indices` select original `.dat` header fields. `field_names` select
loaded field columns by name, including materialized derived fields. Supplying
both selector types to one method raises `ValueError`.

## Layout names

| Name | Shape | Public use |
| --- | --- | --- |
| `udata` | `(nx, ny, nz, nw)` | Preferred user-facing uniform layout |
| `datau` | `(nw, nx, ny, nz)` | Compute-oriented uniform layout |
| `sfc_data` | `(nleafs, nw, bx, by, bz)` | Native block layout |

## Independent tools

### `potential_field_green(b3_bottom, xmin, xmax, nz, *, backend="auto", balance_flux=True)`

Compute a Cartesian potential-field extrapolation from a uniform bottom-face
normal magnetic field.

This helper is independent of AMRVAC files, AMR meshes, dataset objects, and
solver-compatible CT staggered face fields. It accepts a 2D bottom `b3` array on
the lower `z = xmin[2]` face and returns cell-centered magnetic field samples
inside the 3D box.

Return value:

```text
(bfield, geometry)
```

`bfield` shape:

```text
(3, nx, ny, nz)
```

where `nx, ny = b3_bottom.shape`, `bfield[0]` is `b1`, `bfield[1]` is `b2`, and
`bfield[2]` is `b3`.

`geometry` is a frozen `PotentialFieldGeometry` dataclass with `xmin`, `xmax`,
`domain_nx`, `dx`, `dy`, `dz`, `spacing`, `flux_balanced`, and
`removed_flux_mean`. Use `geometry.cell_center_coordinates()` to reconstruct the
1D cell-center coordinate arrays.

Backends:

| Backend | Meaning |
| --- | --- |
| `"auto"` | Use SciPy FFT convolution when available; otherwise use direct NumPy convolution |
| `"fft"` | Require `scipy.signal.fftconvolve` |
| `"direct"` | Use the direct NumPy convolution path |

Example:

```python
import numpy as np
from simesh.tools import potential_field_green

b3_bottom = np.zeros((32, 32))
b3_bottom[12:20, 12:20] = 1.0

bfield, geometry = potential_field_green(
    b3_bottom,
    xmin=[0.0, 0.0, 0.0],
    xmax=[1.0, 1.0, 1.0],
    nz=32,
)

assert bfield.shape == (3, 32, 32, 32)
print(geometry.domain_nx)
```

Helpers:

```python
from simesh.amrvac.layouts import datau_to_udata, udata_to_datau
```
