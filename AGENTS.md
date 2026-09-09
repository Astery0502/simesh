# simesh development

The root package is the selected native AMR analysis implementation, promoted
from `analysis-core/`. Production code, build inputs, tests and examples live
at the repository root. The N4 numerical and ownership boundaries remain fixed.

## Start here

- `README.md` and the user-requested Chinese `docs/application-guide.md`:
  supported scientific workflows, output formats and runnable examples.
- `docs/api-reference.md`: current application signatures and selector rules.
- `docs/interface-review.md`: current interface/resource findings and implemented
  output optimizations, with measured limits.
- `docs/architecture.md`: module responsibilities and runtime dependencies.
- `MIGRATION.md`: import changes and retained Dataset/file interfaces.
- `docs/cython-build.md`: extension layout, build and installation checks.
- `ASSETS.md`: retained algorithm provenance when relevant.

## Code map and editing boundaries

- `src/simesh/`: native Mesh, Source, Fields and scientific consumers.
- `src/simesh/_kernels/`: current Cython consumers; `primitives/` retains
  low-level AMR arithmetic with its own compiler semantics.
- `src/simesh/_amr/`: checked Python wrappers for those primitives.
- `src/simesh/amrvac/`: bundled mutable Dataset and ordinary file interfaces.
  Its private `_mesh/` directory contains the retained stateful Cython mesh,
  forest, Morton code and support headers. Native scientific consumers must
  not depend on its mutable AMRMesh.
- `src/simesh/tools/`: array-only potential field and analytic configurations.
- `legacy/previous/`: archived preceding main package and rewrite provider.
- `legacy/python-first/`: archived earlier Python implementation and pre-package
  sources. Neither archive is part of the active build, import or test surface.
- `legacy/core-development/`: historical comparison runners and artifacts.

New work consumes Source/Fields and existing scientific interfaces. Keep
source/preparation, storage/validity and owned/borrowed lifetime boundaries
explicit. Change core behavior only for a concrete defect or required capability.
Do not restore removed `simesh.utils` or `simesh.legacy` compatibility dispatchers.
Allocated padding is not valid halo. A first derivative consumes one valid halo
layer; region boundaries are not physical boundaries. Numerical schemes and
physical unit/model choices stay explicit. Scientific consumers do not read
Sources or recover missing coverage implicitly.

## Build and validation

- Use root `.venv/bin/python` for local Python commands.
- Install with `.venv/bin/python -m pip install -e '.[dev]'`.
- Run `make build` after changing Cython or compiled module paths.
- Run `make test` for the active package; default collection excludes archives.
- Run representative examples and affected checks for ordinary applications.
  Use an isolated wheel check when changing packaging or source layout.
- Keep old-package comparisons in separate interpreters. Archived virtual
  environments are not portable; recreate them when historical work needs one.

Default to one agent. Use direct source/contract inspection when it settles a
change; avoid tests for descriptive metadata, incidental call order or unchanged
simple formulas. Measure performance only for a concrete question or claim.

## Documentation

Implementation and code comments are in English. The application manual, API
reference and current interface review are in Chinese as explicitly requested
by the user. Keep current documentation under `docs/` small and task-oriented.
Historical development and superseded technical/application documents live in
`legacy/core-development/documentation-before-application-guide/`; they are not
active instructions or prerequisites. Preserve implementation provenance and
license information in ASSETS.md and LICENSE.
