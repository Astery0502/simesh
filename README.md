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

The supported install path is `pyproject.toml`-based. Cython extensions are
compiled automatically during installation.

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

### Using Poetry

```bash
git clone https://github.com/Astery0502/simesh.git
cd simesh
poetry install
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
make build
make build-amr
make test
```

Notes:

- `make build` compiles all `.pyx` files under `src/simesh/utils/lib/`
- `make build-amr` compiles only the `src/simesh/utils/lib/amr/` subtree
- `make test` rebuilds extensions and runs the script-based tests under `tests/`
- tests follow the package structure, for example `tests/utils/lib/` for Cython AMR internals and `tests/amrvac/` for canonical AMRVAC dataset behavior

## Usage

The canonical code path uses `simesh.amrvac` for AMRVAC datasets and
`simesh.utils.lib` for compiled internals.

### Load AMRVAC data

```python
from simesh.amrvac.amrvac_dataset import AMRVACDataSet

ds = AMRVACDataSet(datfile)
ds.load_data()
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
