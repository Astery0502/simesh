# Python API Map

## Purpose

This document identifies the Python modules that are most relevant for user
workflows and how they relate to the lower-level AMR implementation.

## Primary public entrypoints

The clean user-facing API is exported from:

- `src/simesh/amrvac/__init__.py`
- `src/simesh/amrvac/api.py`

The main public functions are:

- `open_dataset(path, *, ghost_width=0)`
- `read_blocks(path, *, field_indices=None, ghost_width=0, include_ghosts=False)`
- `read_uniform(path, *, resolution, bounds=None, field_indices=None, ghost_width=0)`
- `write_datfile(path, output_path, *, field_indices=None, ghost_width=0, overwrite=False)`

The lower-level canonical Python modules live in:

- `src/simesh/amrvac/amrvac_dataset.py`
- `src/simesh/amrvac/datio.py`

Treat `api.py` as the first user-facing entrypoint, `amrvac_dataset.py` as the
stateful dataset implementation, and `datio.py` as the canonical low-level
AMRVAC format module.

## Main objects behind the entrypoints

### Dataset

The canonical dataset object is defined in:

- `src/simesh/amrvac/amrvac_dataset.py`

### Compiled AMR structures

The canonical AMR structures used by `simesh.amrvac` come from:

- `src/simesh/utils/lib/amr/forest.pyx`
- `src/simesh/utils/lib/amr/mesh.pyx`

## Alternate or overlapping API paths

The older Python-first implementation now lives under:

- `src/simesh/legacy/`

This legacy tree is preserved for reference and fallback use, but it is not the
default current API surface.

## Suggested public-API stance

For now, it is safest to think of the public API in two layers:

1. canonical AMRVAC-facing modules under `simesh.amrvac`
2. lower-level Cython acceleration modules under `simesh.utils.lib` that should
   usually stay behind the Python interface

## Array Layouts

The canonical AMRVAC path uses three explicit layout conventions:

- `udata`: user-facing uniform data with shape `(nx, ny, nz, nw)`
- `datau`: compute-oriented uniform data with shape `(nw, nx, ny, nz)`
- `sfc_data`: Morton/SFC block data with shape `(nleafs, nw, bx, by, bz)`

Helpers for converting between `udata` and `datau` live in:

- `src/simesh/amrvac/layouts.py`

## Documentation rule

When adding new user-facing functions:

- document them first in `README.md` if they are high-level
- document them here if they are part of the intended Python API surface
- document implementation details in one of the technical notes under `docs/`
