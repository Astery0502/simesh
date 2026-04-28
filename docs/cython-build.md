# Cython Build Notes

## Purpose

This repository uses Cython to accelerate selected AMR operations. This document
describes where those extensions live and how the current build flow is wired.

## Main build files

- `build.py`
- `setup.py`
- `scripts/build_ext.py`
- `pyproject.toml`

## Current build entrypoint

The packaging configuration uses `setuptools.build_meta`, with `setup.py`
calling into `build.py`.

`build.py`:

- discovers all `.pyx` files recursively under `src/simesh/utils/lib/`
- sets include directories, including NumPy headers
- applies Cython compiler directives
- supports subdirectory-scoped builds for development
- supports `--inplace` builds when run directly
- acts as the single source of truth for setuptools extension discovery

## Discovery model

The package build compiles every `.pyx` file under:

- `src/simesh/utils/lib/`

Files such as `.pxd` are not compiled directly; they act as Cython interface
or header files.

For development, `--group <name>` means:

- compile only the subdirectory `src/simesh/utils/lib/<name>/`

At the moment, the practical example is:

- `--group amr`

## What is compiled

Representative compiled modules:

- `src/simesh/utils/lib/amr/morton.pyx`
- `src/simesh/utils/lib/amr/forest.pyx`
- `src/simesh/utils/lib/amr/mesh.pyx`

These cover:

- Morton encoding and index mappings
- AMR forest construction and connectivity
- mesh indexing and uniform-grid extraction

## Development usage

For local development, the repository includes `scripts/build_ext.py`, which can
invoke `build.py` and optionally clean generated artifacts first.

Typical intent:

- rebuild all compiled extensions in place
- rebuild only one subdirectory during iteration

Current Make targets:

- `make build`
- `make build-amr`
- `make build-amr-openmp`
- `make test`

Editable installs are expected to compile the extensions during:

- `pip install -e .`

After editing `.pyx` files, rebuild in place with:

- `make build`
- `make build-amr`
- `make build-amr-openmp` when testing OpenMP-enabled AMR kernels

To reset generated extension and packaging artifacts:

- `make clean`

Current tests mirror the package structure:

- `tests/utils/lib/` covers compiled AMR internals
- `tests/amrvac/` covers canonical AMRVAC dataset behavior

## Why this matters

The repository contains both canonical and legacy AMR code. When
changing behavior, verify whether the active user workflow depends on:

- canonical AMRVAC code under `src/simesh/amrvac/`
- Cython-backed implementations under `src/simesh/utils/lib/`
- legacy Python-first code under `src/simesh/legacy/`

Do not assume that changing one layer automatically updates the other.

## OpenMP build mode

OpenMP is opt-in. Plain package builds, editable installs, `make build`, and
`make build-amr` do not add OpenMP flags. This keeps the default source build
usable on systems without an OpenMP runtime.

OpenMP can be enabled through either the build flag or the environment variable:

```bash
python build.py --inplace --group amr --openmp
SIMESH_OPENMP=1 pip install -e .
make build-amr-openmp
```

The build helper adds these flags when OpenMP is requested:

- macOS: `-Xpreprocessor -fopenmp` for compilation and `-lomp` for linking
- other platforms: `-fopenmp` for compilation and linking

On macOS, install `libomp` if the OpenMP build cannot find `omp.h` or `libomp`.
Homebrew installs are detected under `/opt/homebrew` and `/usr/local`. If a
user does not have OpenMP available, they should use the default non-OpenMP
build and leave `SIMESH_OPENMP` unset.

The AMR mesh extension exposes build status for user scripts and support
checks:

```python
from simesh.utils import openmp_build_info, openmp_enabled

assert isinstance(openmp_enabled(), bool)
print(openmp_build_info())
```

OpenMP-enabled builds use the same public AMRVAC APIs. Users should tune thread
count with standard OpenMP runtime variables such as `OMP_NUM_THREADS` and
`OMP_DYNAMIC`.

## Caveats

- Editable installs compile the current extensions, but later `.pyx` changes
  still require an explicit rebuild.
- The build configuration should be treated as code to verify against current
  imports, tests, and package structure rather than as guaranteed synchronized
  documentation.
