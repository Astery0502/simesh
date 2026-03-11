# Cython Build Notes

## Purpose

This repository uses Cython to accelerate selected AMR operations. This document
describes where those extensions live and how the current build flow is wired.

## Main build files

- `build.py`
- `scripts/build_ext.py`
- `pyproject.toml`

## Current build entrypoint

The packaging configuration points Poetry at `build.py`.

`build.py`:

- discovers all `.pyx` files recursively under `src/simesh/utils/lib/`
- sets include directories, including NumPy headers
- applies Cython compiler directives
- supports subdirectory-scoped builds for development
- supports `--inplace` builds when run directly

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

## Why this matters

The repository contains both canonical and legacy AMR code. When
changing behavior, verify whether the active user workflow depends on:

- canonical AMRVAC code under `src/simesh/amrvac/`
- Cython-backed implementations under `src/simesh/utils/lib/`
- legacy Python-first code under `src/simesh/legacy/`

Do not assume that changing one layer automatically updates the other.

## Caveats

- Some older packaging files appear to reflect earlier layouts and may not be
  the canonical source of truth for current builds.
- The build configuration should be treated as code to verify against current
  imports, tests, and package structure rather than as guaranteed synchronized
  documentation.
