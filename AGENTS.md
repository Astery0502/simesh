# AGENTS.md

## Purpose

This repository is a Python/Cython toolkit for working with AMRVAC-style
adaptive mesh refinement (AMR) data. The codebase mixes user-facing Python
interfaces with lower-level AMR implementations and compiled extensions.

Use this file as a short navigation and editing aid. Follow a progressive
disclosure path: start here to find the right layer, then go to `README.md`
and the focused documents under `docs/` for detailed mechanics.

## Start Here

- Read `README.md` for the project-level description and current limitations.
- Read `docs/architecture.md` for the overall module layout and intended
  layering.
- Read `docs/python-api-map.md` before changing public interfaces.

## Use These References By Task

- `docs/amrvac-dat-format.md`
  Read this before changing AMRVAC `.dat` parsing, metadata layout, or write
  flow.
- `docs/amr-forest-mesh.md`
  Read this before changing forest reconstruction, connectivity, Morton
  ordering, or ghost-cell behavior.
- `docs/cython-build.md`
  Read this before changing `.pyx` files, compiled module layout, or rebuild
  flow.
- `docs/performance-benchmarks.md`
  Read this before changing benchmark/report tooling or performance workflows.

## Code Map

- `src/simesh/amrvac/`
  Canonical AMRVAC-facing implementation. Keep outer user-facing dataset and
  file-format orchestration here. Primary entrypoints are
  `amrvac_dataset.py` and `datio.py`.
- `src/simesh/utils/lib/`
  Canonical Cython source tree for performance-sensitive AMR internals. `.pyx`
  files here are compiled; `.pxd` files are support headers/interfaces.
- `tests/amrvac/` and `tests/utils/lib/`
  Behavioral reference for the canonical Python and compiled paths. Mirror the
  package structure when adding tests.
- `src/simesh/legacy/`
  Python-first implementation preserved for reference/fallback, not the default
  current path.
- `archive/src_old/`
  Historical code outside the installable package tree.

## Editing Boundaries

- Prefer pure Python changes in `src/simesh/amrvac/` for outer interfaces,
  dataset assembly, and AMRVAC file-format orchestration.
- Prefer Cython changes in `src/simesh/utils/lib/` for performance-sensitive
  AMR internals such as forest structure, Morton ordering, mesh bookkeeping,
  and uniform-grid extraction.
- Do not treat `src/simesh/legacy/` as the default edit target unless the task
  is explicitly about legacy/reference behavior.
- When changing canonical behavior, check whether mirrored coverage belongs in
  `tests/amrvac/` or `tests/utils/lib/`.

## Build And Test Cues

- Supported package/build surface is `pip`/`setuptools` with `Cython`; do not
  assume Poetry, tox, nox, or a pytest-only workflow.
- Use `.venv/bin/python` for local test, build, and ad hoc Python commands.
  The system `python3` may be older than the project's supported Python
  version.
- Use `.venv/bin/python -m pip install -e .` for editable local development.
- Use `make build` to rebuild all compiled extensions after editing `.pyx`
  files.
- Use `make build-amr` when iterating only on `src/simesh/utils/lib/amr/`.
- Use `make test` for the repo's current scripted test flow.
- Use `make clean` to remove compiled extensions and generated build artifacts.
- `scripts/build_ext.py` is the development helper behind the in-place rebuild
  flow.

## Project Constraints

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
