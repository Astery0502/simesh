# Python API Map

## Purpose

This document identifies the Python modules that are most relevant for user
workflows and how they relate to the lower-level AMR implementation.

## Primary public entrypoints

The most important canonical Python modules currently live in:

- `src/simesh/amrvac/amrvac_dataset.py`
- `src/simesh/amrvac/datio.py`

Treat `amrvac_dataset.py` as the canonical dataset entrypoint and `datio.py` as
the canonical low-level AMRVAC format module.

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

## Documentation rule

When adding new user-facing functions:

- document them first in `README.md` if they are high-level
- document them here if they are part of the intended Python API surface
- document implementation details in one of the technical notes under `docs/`
