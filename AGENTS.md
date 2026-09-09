# simesh development

The root package is the selected native AMR analysis implementation, promoted
from `analysis-core/`. Production code, build inputs, tests and examples live
at the repository root. The N4 numerical and ownership boundaries remain fixed.

## Start here

- `README.md`: supported scientific workflows, output formats and runnable examples.
- `docs/user/index.md` and `docs/user/api.md`: concise user entry and generated public interfaces.
- `docs/dev/index.md` and `docs/dev/api.md`: developer entry and core/advanced interfaces.
- `docs/index.md`: documentation map, separating application use from development.
- `docs/dev/architecture.md`: module responsibilities and runtime dependencies.
- `docs/dev/cython-build.md`: extension layout, build and installation checks.
- `ASSETS.md`: retained algorithm provenance when relevant.

## Code map and editing boundaries

- `src/simesh/`: native Mesh, Source, Fields and scientific consumers.
- `src/simesh/_kernels/`: current Cython consumers; `primitives/` retains
  low-level AMR arithmetic with its own compiler semantics.
- `src/simesh/_amr/`: checked Python wrappers for those primitives.
- `src/simesh/io/`: native Source adapters and ordinary AMRVAC v5 file products.
  Its private `_v5/` directory owns file indexing, reading and serialization.
  Mutable Dataset and AMRMesh implementations remain historical and are not shipped.
- `src/simesh/tools/`: array-only potential field and analytic configurations.
- `legacy/previous/`: archived preceding main package and rewrite provider.
- `legacy/python-first/`: archived earlier Python implementation and pre-package
  sources. Neither archive is part of the active build, import or test surface.
- `legacy/core-development/`: historical comparison runners and artifacts.

New work consumes Source/Fields and existing scientific interfaces. Keep
source/preparation, storage/validity and owned/borrowed lifetime boundaries
explicit. Change core behavior only for a concrete defect or required capability.
Do not restore removed `simesh.amrvac`, `simesh.utils` or `simesh.legacy`
compatibility dispatchers.
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

Interface descriptions live in English NumPy-style source docstrings. API pages
under `docs/user/` and `docs/dev/` contain explicit mkdocstrings object references,
not copied signatures. Install `.[docs]`; run `make docs` for static API/parameter
checks and a strict HTML build, or `make docs-serve` for local preview. Do not
commit `site/`. Keep exact API descriptions at one source definition and retain
ownership, support, units and termination constraints when shortening text.
Follow the content placement and Notes brevity rules in
[Docstring content](docs/dev/cython-build.md#docstring-content); that section is
the authoritative writing guide rather than a second copy of each API contract.

Implementation and code comments are in English. Active Chinese documentation
is limited to application use. Keep
architecture, build and source-generated API references in the English developer
and user sections. Put development reviews, measurements and future test plans
under `legacy/core-development/`, outside the active documentation site. Keep current documentation under `docs/` small and task-oriented.
Do not retain superseded development or migration documents in the repository.
Preserve implementation provenance and license information in ASSETS.md and LICENSE.
