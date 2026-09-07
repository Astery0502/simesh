# Documentation Index

## User docs

Read these when you want to use `simesh` from Python.

| Need | Read |
| --- | --- |
| Install the package or set up development builds | `docs/installation.md` |
| Choose the right public interface | `docs/user-guide.md` |
| Work with Cartesian 2D data | `docs/2d-guide.md` |
| Understand potential-field extrapolation equations | `docs/potential-field-tools.md` |
| Look up public function signatures | `docs/api-reference.md` |

## Maintainer docs

Project-level future analysis-core specifications have their own
[entry point](analysis-core/README.md), independent of the rewrite's historical
implementation tree. Start with [prepared fields](analysis-core/prepared-fields.md)
for the paired ghost-preparation and data-consumption requirements.
For ongoing analysis-core development, read the
[active checkpoint](analysis-core/current.md) and
[development progression](analysis-core/development.md); follow their links for
the selected design, exploration and acceptance evidence.

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
