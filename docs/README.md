# Documentation Index

## User docs

Read these when you want to use `simesh` from Python.

| Need | Read |
| --- | --- |
| Install the package or set up development builds | `docs/installation.md` |
| Choose the right public interface | `docs/user-guide.md` |
| Work with Cartesian 2D data | `docs/2d-guide.md` |
| Look up public function signatures | `docs/api-reference.md` |

## Maintainer docs

Read these when you are changing implementation behavior.

| Need | Read |
| --- | --- |
| Understand the intended public API surface | `docs/python-api-map.md` |
| Understand repository layout and layers | `docs/architecture.md` |
| Change AMRVAC `.dat` parsing or writing | `docs/amrvac-dat-format.md` |
| Change AMR forest, mesh, Morton order, or ghost cells | `docs/amr-forest-mesh.md` |
| Change Cython build behavior | `docs/cython-build.md` |
| Run or update performance workflows | `docs/performance-benchmarks.md` |

## Documentation roles

- `README.md` is the project landing page.
- `docs/user-guide.md` explains workflows and tradeoffs.
- `docs/api-reference.md` lists public functions and compact examples.
- Technical implementation notes stay in focused `docs/` pages instead of the
  README.
