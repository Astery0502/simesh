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
- `make test`

Editable installs are expected to compile the extensions during:

- `pip install -e .`

After editing `.pyx` files, rebuild in place with:

- `make build`
- `make build-amr`

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

## Caveats

- Editable installs compile the current extensions, but later `.pyx` changes
  still require an explicit rebuild.
- The build configuration should be treated as code to verify against current
  imports, tests, and package structure rather than as guaranteed synchronized
  documentation.
