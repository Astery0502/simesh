# AGENTS.md

## Purpose

This repository is a Python/Cython toolkit for working with AMRVAC-style
adaptive mesh refinement (AMR) data. The codebase mixes user-facing Python
interfaces with lower-level AMR implementations and compiled extensions.

Use this file as a navigation aid, not as the primary technical reference.
Detailed explanations live in `docs/`.

## Start Here

- Read `README.md` for the project-level description and current limitations.
- Read `docs/architecture.md` for the overall module layout and intended
  layering.
- Read `docs/python-api-map.md` before changing public interfaces.

## Canonical Technical References

- `docs/amrvac-dat-format.md`
  AMRVAC `.dat` layout, read/write flow, and which modules own file parsing.
- `docs/amr-forest-mesh.md`
  Forest, octree, Morton ordering, connectivity, and ghost-cell handling.
- `docs/cython-build.md`
  How Cython extensions are organized and built in development and packaging.

## Code Map

- `src/simesh/amrvac/`
  Canonical AMRVAC-facing implementation.
- `src/simesh/utils/lib/`
  Cython source tree; `.pyx` files here are compiled, `.pxd` files are support
  headers/interfaces.
- `src/simesh/legacy/`
  Python-first implementation preserved for reference/fallback, not the default
  current path.
- `archive/src_old/`
  Historical code outside the installable package tree.
- `tests/`
  Behavioral reference for current expectations and supported workflows.

## Working Assumptions

- The current implementation is primarily targeted at Cartesian 3D AMR data.
- Public user workflows are centered on `simesh.amrvac` and `simesh.utils`.
- There are overlapping Python-first and Cython-backed implementations in the
  repository. The Python-first path now lives under `simesh.legacy`; do not
  treat it as the default current implementation.
- Default build behavior compiles all `.pyx` files under `src/simesh/utils/lib/`.

## Documentation Maintenance

- Put project-level summaries in `README.md`.
- Put design and implementation notes in `docs/`.
- Keep this file short and operational.
- When adding a major subsystem, add one focused document in `docs/` and link
  it here instead of expanding this file into a full manual.
