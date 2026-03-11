# Architecture

## Project goal

`simesh` is built to read, write, explore, and manipulate AMRVAC-style adaptive
mesh refinement data. In practice, that means the repository serves two roles:

- a file-format toolkit for AMRVAC `.dat` snapshots
- an AMR mesh toolkit for block-structured simulation data

The codebase is split between Python interfaces for users and Cython-backed
implementations for speed-sensitive AMR operations.

## High-level layers

### 1. User-facing Python entrypoints

The canonical user workflows live under `src/simesh/amrvac/`.

Important functions:

- `amr_loader(...)`
- `load_from_uarrays(...)`
- `write_dat_metadata(...)`
- `write_dat_field(...)`

These functions are the most direct expression of the intended public API:
users load AMRVAC files, construct datasets from arrays, inspect metadata, and
write data back out in AMRVAC-compatible form.

### 2. Dataset and mesh objects

The Python object model is built around:

- dataset and format-facing objects in `src/simesh/amrvac/`
- compiled AMR structures and support code in `src/simesh/utils/`

These modules handle:

- dataset assembly from parsed file components
- block coordinate bookkeeping
- neighbor relationships
- ghost-cell and boundary updates
- block-level AMR operations

### 3. Cython performance layer

Compiled AMR structures live under `src/simesh/utils/lib/`, with the current
active AMR-specific implementations under `src/simesh/utils/lib/amr/`.

These modules implement performance-critical pieces such as:

- Morton ordering
- forest construction and connectivity
- mesh indexing and coordinate bookkeeping
- uniform-grid extraction from AMR data

The intent is that Python exposes convenient interfaces while Cython handles
the heavy loops and memory-sensitive logic.

## Main directories

### `src/simesh/amrvac/`

Canonical AMRVAC-specific file I/O and dataset code.

### `src/simesh/legacy/`

Legacy Python-first implementation path, including the older mesh, dataset,
geometry, and frontend code that is preserved for reference and fallback use.

### `src/simesh/utils/lib/`

Cython source tree used for speed-sensitive operations. All `.pyx` files under
this directory are part of the compiled extension discovery path. The current
AMR-focused modules live under `src/simesh/utils/lib/amr/`.

### `src/simesh/utils/configurations.py`

Helpers for generating synthetic or physically motivated fields, useful for
testing and constructing example datasets.

## Current architectural reality

The repository currently contains two implementation tracks:

- the canonical path centered on `src/simesh/amrvac/` and `src/simesh/utils/`
- a legacy Python-first path centered on `src/simesh/legacy/`

This is important when making changes. Work on the canonical path by default and
treat `simesh.legacy` as preserved older code rather than the primary target.

## Recommended reading order

1. `README.md`
2. `docs/python-api-map.md`
3. `docs/amrvac-dat-format.md`
4. `docs/amr-forest-mesh.md`
5. `docs/cython-build.md`
