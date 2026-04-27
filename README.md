# simesh

`simesh` is a Python/Cython toolkit for working with AMRVAC-style adaptive mesh
refinement (AMR) data. The current canonical implementation is centered on
`src/simesh/amrvac/` and `src/simesh/utils/`.

The project follows a two-layer design:

- Python provides the user-facing interfaces for loading datasets, creating new
  datasets from NumPy arrays, inspecting metadata, and exporting results.
- Cython-backed AMR data structures and kernels provide the fast path for
  forest connectivity, Morton ordering, mesh bookkeeping, and uniform-grid
  extraction.

In practice, the package is meant to serve as both:

- an AMRVAC `.dat` reader/writer
- an AMR mesh manipulation library for block-structured simulation data

## Core capabilities

- Load AMRVAC `.dat` files into Python-accessible dataset and mesh objects
- Parse and write AMRVAC metadata, forest structure, tree information, and
  block field data
- Represent the AMR hierarchy through octree/forest-based data structures
- Manipulate block data with ghost-cell handling and AMR neighbor connectivity
- Build datasets directly from uniform NumPy arrays without an existing input
  file
- Resample AMR data onto uniform grids for analysis or downstream workflows
- Export AMR data to VTK hierarchical box output

## Architecture

- `src/simesh/amrvac/`: canonical AMRVAC-facing dataset and file-format modules
- `src/simesh/utils/lib/`: Cython source tree; all `.pyx` files under this
  directory are compiled during package build, with `.pxd` files used as
  headers/interfaces
- `src/simesh/utils/configurations.py`: utilities for generating synthetic or
  physically motivated field configurations
- `src/simesh/legacy/`: preserved Python-first implementation path kept as
  legacy/reference
- `archive/src_old/`: archived historical code kept outside the installable
  package tree

## Installation

The supported install path is `pip`/`setuptools`-based. Cython extensions are
compiled automatically during installation, including editable installs.

### Using pip

```bash
git clone https://github.com/Astery0502/simesh.git
cd simesh
pip install .
```

For editable local development:

```bash
pip install -e .
```

#### Requirements

- Python ≥ 3.11
- NumPy ≥ 1.23.5
- Cython ≥ 3.0 for source builds
- JupyterLab and ipykernel are optional development dependencies

## Development build and test

Package installation compiles all Cython modules found under
`src/simesh/utils/lib/`.

For local development:

```bash
pip install -e .
make build
make build-amr
make test
make benchmark-smoke
```

Notes:

- `make build` compiles all `.pyx` files under `src/simesh/utils/lib/`
- `make build-amr` compiles only the `src/simesh/utils/lib/amr/` subtree
- after editing `.pyx` files, rerun `make build` to rebuild extensions in place
- `make clean` removes compiled extensions and generated packaging artifacts
- `make test` rebuilds extensions and runs the script-based tests under `tests/`
- tests follow the package structure, for example `tests/utils/lib/` for Cython AMR internals and `tests/amrvac/` for canonical AMRVAC dataset behavior
- `make benchmark-smoke` runs a tiny AMRVAC performance-report smoke test; see
  `docs/performance-benchmarks.md` for full scaling reports

## Usage

The canonical code path uses `simesh.amrvac` for AMRVAC datasets and
`simesh.utils.lib` for compiled internals.

### Read AMRVAC data

```python
from simesh.amrvac import read_blocks, read_uniform

blocks = read_blocks(datfile, field_indices=[0, 1])
ghosted = read_blocks(datfile, field_indices=[0, 1], ghost_width=2, include_ghosts=True)
grid = read_uniform(datfile, resolution=(128, 128, 128), field_indices=[0])
smooth_grid = read_uniform(
    datfile,
    resolution=(128, 128, 128),
    field_indices=[0],
    ghost_width=1,
    interpolation="linear",
)
```

Choose the interpolation mode by intent and memory budget:

- `interpolation="zero"` is the default low-memory path. It is exact for
  native level-1 uniform data and remains useful for AMR data when you do not
  want to allocate ghost-cell storage.
- `interpolation="linear"` performs trilinear resampling through the canonical
  Cython ghost-cell path. Use it when you want smoother uniform-grid sampling
  across AMR blocks and can afford `ghost_width > 0`.
- For purely uniform level-1 data sampled on its native full-domain grid, prefer
  the exact full-grid path through `open_dataset(...).uniform_full()`.

Use `open_dataset()` when you need a stateful object, for example to mutate
block data and refresh ghost cells before writing:

```python
from simesh.amrvac import open_dataset

ds = open_dataset(datfile, ghost_width=2)
ds.load_data(field_indices=[0, 1])
ds.blocks()[:, 0] *= 1.01
ds.exchange_ghost_cells()
ds.write_datfile("updated.dat")
```

### Low-level metadata access

```python
from simesh.amrvac.datio import get_metadata

header, forest, tree = get_metadata(datfile)
```

### Array Layouts

The canonical AMRVAC path uses three array layout names:

- `udata`: user-facing uniform data with shape `(nx, ny, nz, nw)`
- `datau`: compute-oriented uniform data with shape `(nw, nx, ny, nz)`
- `sfc_data`: Morton/SFC block data with shape `(nleafs, nw, bx, by, bz)`

Conversion helpers live in `simesh.amrvac.layouts`:

```python
from simesh.amrvac.layouts import datau_to_udata, udata_to_datau
```

Legacy Python-only code is preserved under `simesh.legacy`, but it is not the
default path and is not part of the current default test workflow.

## Limitations

The current implementation is primarily targeted at ***Cartesian 3D AMR meshes***.
Several code paths and tests also assume constant or simple physical boundary
handling. If you need broader geometry or dimensional support, treat the current
state as specialized rather than fully general.
